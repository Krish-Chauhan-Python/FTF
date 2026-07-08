import argparse
import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

ARM_JOINTS = [name for name in MOTORS if name != "gripper"]


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_snapshots(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected list in {path}, got {type(data)!r}")
    return data


def get_positions_from_snapshot(snapshot: dict[str, object]) -> dict[str, float]:
    positions = snapshot.get("positions")
    if not isinstance(positions, dict):
        raise TypeError(f"Invalid snapshot positions payload: {type(positions)!r}")

    result: dict[str, float] = {}
    for motor_name in MOTORS:
        if motor_name in positions:
            result[motor_name] = float(positions[motor_name])
    return result


def read_gripper_load_abs(bus: FeetechMotorsBus) -> int:
    raw_load = bus.read("Present_Load", "gripper", normalize=False)
    return abs(int(raw_load))


def write_full_goal(bus: FeetechMotorsBus, arm_positions: dict[str, float], gripper_goal: float) -> None:
    goals: dict[str, float] = {joint: arm_positions[joint] for joint in ARM_JOINTS if joint in arm_positions}
    goals["gripper"] = gripper_goal
    bus.sync_write("Goal_Position", goals, normalize=False)


def move_to_pose_fixed_gripper(
    bus: FeetechMotorsBus,
    target_arm_pose: dict[str, float],
    gripper_goal: float,
    transition_steps: int,
    transition_step_delay: float,
) -> None:
    current_raw = bus.sync_read("Present_Position", normalize=False)
    start_arm_pose = {joint: float(current_raw[joint]) for joint in ARM_JOINTS}

    for step in range(1, max(1, transition_steps) + 1):
        t = smoothstep(step / max(1, transition_steps))
        interp: dict[str, float] = {}
        for joint in ARM_JOINTS:
            start_val = start_arm_pose[joint]
            end_val = target_arm_pose.get(joint, start_val)
            interp[joint] = start_val + (end_val - start_val) * t

        write_full_goal(bus, interp, gripper_goal)
        time.sleep(max(0.0, transition_step_delay))


def close_gripper_until_load(
    bus: FeetechMotorsBus,
    initial_gripper_goal: float,
    target_load: int,
    close_step: float,
    min_gripper_goal: float,
    poll_delay: float,
) -> tuple[float, int]:
    gripper_goal = float(initial_gripper_goal)
    bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)

    load_abs = read_gripper_load_abs(bus)
    while load_abs < target_load and gripper_goal > min_gripper_goal:
        gripper_goal = max(min_gripper_goal, gripper_goal - close_step)
        bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)
        time.sleep(max(0.0, poll_delay))
        load_abs = read_gripper_load_abs(bus)
        print(
            f"\r[LOAD SEEK] target={target_load} current={load_abs} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )

    print()
    return gripper_goal, load_abs


def transition_with_constant_load_control(
    bus: FeetechMotorsBus,
    start_pose: dict[str, float],
    end_pose: dict[str, float],
    gripper_goal: float,
    target_load: int,
    load_deadband: int,
    gripper_adjust_step: float,
    min_gripper_goal: float,
    max_gripper_goal: float,
    transition_steps: int,
    transition_step_delay: float,
) -> float:
    steps = max(1, transition_steps)
    for step in range(1, steps + 1):
        t = smoothstep(step / steps)
        arm_interp: dict[str, float] = {}
        for joint in ARM_JOINTS:
            start_val = start_pose.get(joint, 0.0)
            end_val = end_pose.get(joint, start_val)
            arm_interp[joint] = start_val + (end_val - start_val) * t

        load_abs = read_gripper_load_abs(bus)
        if load_abs < target_load - load_deadband:
            gripper_goal -= gripper_adjust_step
        elif load_abs > target_load + load_deadband:
            gripper_goal += gripper_adjust_step

        gripper_goal = clamp(gripper_goal, min_gripper_goal, max_gripper_goal)
        write_full_goal(bus, arm_interp, gripper_goal)

        print(
            f"\r[TRANSITION] step={step}/{steps} load={load_abs} target={target_load} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )
        time.sleep(max(0.0, transition_step_delay))

    print()
    return gripper_goal


def ramp_gripper_to_goal(
    bus: FeetechMotorsBus,
    start_goal: float,
    end_goal: float,
    ramp_steps: int,
    ramp_step_delay: float,
) -> float:
    steps = max(1, int(ramp_steps))
    delay = max(0.0, float(ramp_step_delay))

    for step in range(1, steps + 1):
        t = smoothstep(step / steps)
        gripper_goal = start_goal + (end_goal - start_goal) * t
        bus.write("Goal_Position", "gripper", gripper_goal, normalize=False)
        print(
            f"\r[FINAL RAMP] step={step}/{steps} gripper_goal={gripper_goal:.1f}",
            end="",
            flush=True,
        )
        time.sleep(delay)

    print()
    return end_goal


def apply_gripper_limits(bus: FeetechMotorsBus, max_torque_limit: int, protection_current: int) -> None:
    bus.write("Max_Torque_Limit", "gripper", max(0, int(max_torque_limit)), normalize=False)
    bus.write("Protection_Current", "gripper", max(0, int(protection_current)), normalize=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run enter-gated 5-pose sequence with gripper load control between poses"
    )
    parser.add_argument("--port", type=str, default="COM5", help="Serial port for SO101 controller")
    parser.add_argument("--readings-file", type=str, default="servo_readings.json", help="Path to saved snapshots JSON")
    parser.add_argument("--target-load", type=int, default=200, help="Absolute Present_Load setpoint to maintain")
    parser.add_argument("--load-deadband", type=int, default=5, help="Allowed load error band before adjusting gripper")
    parser.add_argument("--initial-gripper-goal", type=float, default=3000.0, help="Initial gripper goal position")
    parser.add_argument("--final-gripper-goal", type=float, default=3000.0, help="Final gripper goal at end of sequence")
    parser.add_argument("--close-step", type=float, default=10.0, help="Amount to decrease gripper per load-seek step")
    parser.add_argument("--gripper-adjust-step", type=float, default=4.0, help="Adjustment step while maintaining load")
    parser.add_argument("--min-gripper-goal", type=float, default=0.0, help="Minimum gripper goal clamp")
    parser.add_argument("--max-gripper-goal", type=float, default=4095.0, help="Maximum gripper goal clamp")
    parser.add_argument("--transition-steps", type=int, default=60, help="Interpolation steps between poses")
    parser.add_argument("--transition-step-delay", type=float, default=0.02, help="Delay per interpolation step")
    parser.add_argument("--poll-delay", type=float, default=0.03, help="Delay during gripper load seek")
    parser.add_argument("--final-ramp-steps", type=int, default=80, help="Interpolation steps for final gripper ramp")
    parser.add_argument(
        "--final-ramp-step-delay",
        type=float,
        default=0.02,
        help="Delay per step during final gripper ramp",
    )
    parser.add_argument("--gripper-max-torque-limit", type=int, default=800, help="Raw Max_Torque_Limit for gripper")
    parser.add_argument("--gripper-protection-current", type=int, default=400, help="Raw Protection_Current for gripper")
    args = parser.parse_args()

    snapshots = load_snapshots(Path(args.readings_file))
    if len(snapshots) < 5:
        raise ValueError("Need at least 5 snapshots in the JSON file")

    poses = [get_positions_from_snapshot(snapshots[i]) for i in range(5)]

    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)

    try:
        bus.connect(handshake=False)
        bus.enable_torque()
        apply_gripper_limits(bus, args.gripper_max_torque_limit, args.gripper_protection_current)

        initial_goal = clamp(args.initial_gripper_goal, args.min_gripper_goal, args.max_gripper_goal)
        print("[STEP] Moving to position 1 with gripper set to initial goal")
        move_to_pose_fixed_gripper(
            bus,
            target_arm_pose=poses[0],
            gripper_goal=initial_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
        )

        input("[INPUT] Press Enter to start closing gripper until load setpoint is reached...")
        gripper_goal, reached_load = close_gripper_until_load(
            bus,
            initial_gripper_goal=initial_goal,
            target_load=args.target_load,
            close_step=args.close_step,
            min_gripper_goal=args.min_gripper_goal,
            poll_delay=args.poll_delay,
        )
        print(f"[INFO] Load seek done: load={reached_load}, gripper_goal={gripper_goal:.1f}")

        input("[INPUT] Press Enter to move to position 2...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[0],
            end_pose=poses[1],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
        )

        input("[INPUT] Press Enter to move to position 3...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[1],
            end_pose=poses[2],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
        )

        input("[INPUT] Press Enter to move to position 4...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[2],
            end_pose=poses[3],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
        )

        input("[INPUT] Press Enter to move to position 5...")
        gripper_goal = transition_with_constant_load_control(
            bus,
            start_pose=poses[3],
            end_pose=poses[4],
            gripper_goal=gripper_goal,
            target_load=args.target_load,
            load_deadband=args.load_deadband,
            gripper_adjust_step=args.gripper_adjust_step,
            min_gripper_goal=args.min_gripper_goal,
            max_gripper_goal=args.max_gripper_goal,
            transition_steps=args.transition_steps,
            transition_step_delay=args.transition_step_delay,
        )

        final_goal = clamp(args.final_gripper_goal, args.min_gripper_goal, args.max_gripper_goal)
        gripper_goal = ramp_gripper_to_goal(
            bus,
            start_goal=gripper_goal,
            end_goal=final_goal,
            ramp_steps=args.final_ramp_steps,
            ramp_step_delay=args.final_ramp_step_delay,
        )
        print(f"[DONE] Position 5 reached. Gripper slowly ramped to {gripper_goal:.1f}")

    finally:
        if bus.is_connected:
            # Avoid teardown failure when gripper reports overload status during torque-disable writes.
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
