#!/usr/bin/env python3
"""
Automatic Terminal Corruption Detection & Monitoring Script
Continuously monitors terminal for corruption bugs and logs detailed diagnostics
"""

import datetime
import json
import re
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CorruptionEvent:
    timestamp: str
    command_sent: str
    command_received: str
    corruption_type: str
    details: dict
    stage: str  # 'encode', 'transmission', 'decode'


class TerminalCorruptionMonitor:
    def __init__(self, terminal_instance, log_dir="./terminal_logs"):
        self.terminal = terminal_instance
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Monitoring state
        self.corruption_events = []
        self.test_counter = 0
        self.is_monitoring = False
        self.monitoring_thread = None

        # Setup logging
        self.setup_logging()

        # Patch terminal methods for monitoring
        self.patch_terminal_methods()

        # Test command database
        self.test_commands = self.generate_test_commands()

    def setup_logging(self):
        """Setup detailed logging system"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"terminal_debug_{timestamp}.log"
        self.corruption_log = self.log_dir / f"corruption_events_{timestamp}.json"

        print(f"📝 Logging to: {self.log_file}")
        print(f"🚨 Corruption events: {self.corruption_log}")

    def log(self, level: str, message: str, data: Optional[dict] = None):
        """Unified logging function"""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"

        if data:
            log_entry += f" | Data: {json.dumps(data)}"

        print(log_entry)

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def generate_test_commands(self) -> list[dict]:
        """Generate comprehensive test command database"""
        commands = [
            # Basic commands
            {"cmd": "echo 'hello'", "expected_in_output": "hello", "category": "basic"},
            {"cmd": "pwd", "expected_in_output": None, "category": "basic"},
            {"cmd": "ls", "expected_in_output": None, "category": "basic"},
            # Problematic terms from logs
            {"cmd": "echo 'astropy'", "expected_in_output": "astropy", "category": "problematic"},
            {"cmd": "echo 'mask'", "expected_in_output": "mask", "category": "problematic"},
            {"cmd": "echo 'handle_mask'", "expected_in_output": "handle_mask", "category": "problematic"},
            {"cmd": "echo 'array'", "expected_in_output": "array", "category": "problematic"},
            {"cmd": "echo 'nddata'", "expected_in_output": "nddata", "category": "problematic"},
            # Python imports that were problematic
            {
                "cmd": "python -c \"import numpy; print('numpy ok')\"",
                "expected_in_output": "numpy ok",
                "category": "python",
            },
            {
                "cmd": "python -c \"print('astropy.nddata')\"",
                "expected_in_output": "astropy.nddata",
                "category": "python",
            },
            {
                "cmd": "python -c \"print('handle_mask=np.bitwise_or')\"",
                "expected_in_output": "handle_mask=np.bitwise_or",
                "category": "python",
            },
            # Special characters and edge cases
            {"cmd": "echo 'test=value'", "expected_in_output": "test=value", "category": "special"},
            {"cmd": "echo 'file.extension'", "expected_in_output": "file.extension", "category": "special"},
            {"cmd": "echo 'under_score'", "expected_in_output": "under_score", "category": "special"},
            {"cmd": "echo 'multiple words'", "expected_in_output": "multiple words", "category": "special"},
            # Long commands
            {
                "cmd": "echo 'this_is_a_very_long_command_that_might_trigger_buffer_issues'",
                "expected_in_output": "this_is_a_very_long_command_that_might_trigger_buffer_issues",
                "category": "long",
            },
            # Control character tests
            {"cmd": "echo 'no_control_chars'", "expected_in_output": "no_control_chars", "category": "control"},
        ]

        # Add stress test variations
        stress_terms = ["astropy", "mask", "handle", "array", "bitwise"]
        for term in stress_terms:
            for i in range(3):
                commands.append(
                    {"cmd": f"echo '{term}_{i}'", "expected_in_output": f"{term}_{i}", "category": "stress"}
                )

        return commands

    def patch_terminal_methods(self):
        """Patch terminal methods to intercept and monitor calls"""
        # Store originals
        self.original_encode_input = self.terminal.encode_input
        self.original_call = self.terminal.__call__

        # Patch encode_input
        def monitored_encode_input(text: str) -> bytes:
            result = self.original_encode_input(text)
            self.check_encode_corruption(text, result)
            return result

        # Patch __call__
        def monitored_call(input_text: str, timeout=1):
            start_time = time.time()
            try:
                result = self.original_call(input_text, timeout)
                execution_time = time.time() - start_time
                self.check_transmission_corruption(input_text, result, execution_time)
                return result
            except Exception as e:
                self.log(
                    "ERROR",
                    f"Terminal call failed: {e}",
                    {"input": input_text, "exception": str(e), "traceback": traceback.format_exc()},
                )
                raise

        # Apply patches
        self.terminal.encode_input = monitored_encode_input
        self.terminal.__call__ = monitored_call

    def check_encode_corruption(self, original: str, encoded: bytes):
        """Check for corruption in encode_input step"""
        try:
            decoded_back = encoded.decode("utf-8")
            if original != decoded_back:
                event = CorruptionEvent(
                    timestamp=datetime.datetime.now().isoformat(),
                    command_sent=original,
                    command_received=decoded_back,
                    corruption_type="encode_mismatch",
                    details={"original_bytes": len(original.encode("utf-8")), "encoded_bytes": len(encoded)},
                    stage="encode",
                )
                self.record_corruption(event)
        except Exception as e:
            self.log("ERROR", f"Error in encode corruption check: {e}")

    def check_transmission_corruption(self, input_text: str, output: str, execution_time: float):
        """Check for corruption in transmission/execution"""
        # Look for echo patterns to detect what was actually sent
        echo_patterns = [
            r"echo ['\"](.+?)['\"]",
            r"print\(['\"](.+?)['\"]",
        ]

        corruption_detected = False
        details = {"input_length": len(input_text), "output_length": len(output), "execution_time": execution_time}

        # Check for common corruption patterns
        for pattern_name, pattern in [
            ("double_chars", r"(\w)\1{2,}"),  # Three or more repeated chars
            ("missing_chars", r"[^\w\s]"),  # Special chars that might get lost
        ]:
            matches = re.findall(pattern, output)
            if matches:
                details[f"{pattern_name}_matches"] = matches

        # Check for specific echo command corruption
        if "echo" in input_text:
            for pattern in echo_patterns:
                match = re.search(pattern, input_text)
                if match:
                    expected_text = match.group(1)
                    if expected_text not in output:
                        # Check for corrupted version in output
                        output_words = output.split()
                        for word in output_words:
                            if self.is_corrupted_version(expected_text, word):
                                event = CorruptionEvent(
                                    timestamp=datetime.datetime.now().isoformat(),
                                    command_sent=expected_text,
                                    command_received=word,
                                    corruption_type="echo_corruption",
                                    details=details,
                                    stage="transmission",
                                )
                                self.record_corruption(event)
                                corruption_detected = True
                                break

        # Log clean executions too (for baseline)
        if not corruption_detected:
            self.log(
                "DEBUG",
                "Clean execution",
                {
                    "input": input_text[:50] + "..." if len(input_text) > 50 else input_text,
                    "execution_time": execution_time,
                },
            )

    def is_corrupted_version(self, original: str, candidate: str) -> bool:
        """Check if candidate is a corrupted version of original"""
        # Simple heuristics for common corruption patterns

        # Check for double character insertions
        if len(candidate) > len(original):
            # Remove double chars and see if it matches
            dedoubled = re.sub(r"(\w)\1+", r"\1", candidate)
            if dedoubled == original:
                return True

        # Check for character substitutions (max 30% different)
        if len(candidate) == len(original):
            differences = sum(1 for a, b in zip(original, candidate, strict=False) if a != b)
            if 0 < differences <= len(original) * 0.3:
                return True

        # Check for insertions/deletions (Levenshtein-like)
        if abs(len(candidate) - len(original)) <= 3:
            # Simple edit distance check
            if original in candidate or candidate in original:
                return True

        return False

    def record_corruption(self, event: CorruptionEvent):
        """Record a corruption event"""
        self.corruption_events.append(event)

        # Log immediately
        self.log(
            "CORRUPTION",
            f"Detected {event.corruption_type}",
            {"sent": event.command_sent, "received": event.command_received, "stage": event.stage},
        )

        # Save to JSON log
        with open(self.corruption_log, "w") as f:
            json.dump([asdict(event) for event in self.corruption_events], f, indent=2)

        print(f"🚨 CORRUPTION DETECTED #{len(self.corruption_events)}: {event.corruption_type}")
        print(f"   Sent: '{event.command_sent}'")
        print(f"   Got:  '{event.command_received}'")

    def run_single_test(self, test_cmd: dict) -> bool:
        """Run a single test command and check for corruption"""
        self.test_counter += 1
        cmd = test_cmd["cmd"]

        try:
            start_time = time.time()
            result = self.terminal(cmd + "\n", timeout=3)
            execution_time = time.time() - start_time

            # Check if expected output is present
            expected = test_cmd.get("expected_in_output")
            if expected and expected not in result:
                # Might be corruption, check more carefully
                self.check_transmission_corruption(cmd, result, execution_time)
                return False

            # Log successful test
            if self.test_counter % 10 == 0:  # Log every 10th test
                self.log(
                    "INFO",
                    f"Test #{self.test_counter} passed",
                    {"command": cmd, "category": test_cmd["category"], "execution_time": execution_time},
                )

            return True

        except Exception as e:
            self.log(
                "ERROR",
                f"Test #{self.test_counter} failed with exception",
                {"command": cmd, "exception": str(e), "traceback": traceback.format_exc()},
            )
            return False

    def run_test_suite(self, iterations: int = 1):
        """Run the full test suite"""
        print(f"🧪 Running test suite with {iterations} iterations...")
        print(f"   Total commands to test: {len(self.test_commands) * iterations}")

        start_time = time.time()
        failed_tests = 0
        total_tests = 0

        for iteration in range(iterations):
            print(f"\n--- Iteration {iteration + 1}/{iterations} ---")

            for test_cmd in self.test_commands:
                total_tests += 1
                success = self.run_single_test(test_cmd)
                if not success:
                    failed_tests += 1

                # Small delay to prevent overwhelming the terminal
                time.sleep(0.1)

                # Progress update every 25 tests
                if total_tests % 25 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"   Progress: {total_tests}/{len(self.test_commands) * iterations} "
                        f"(Failed: {failed_tests}, Time: {elapsed:.1f}s)"
                    )

        # Final report
        elapsed_time = time.time() - start_time
        success_rate = ((total_tests - failed_tests) / total_tests) * 100

        print("\n📊 TEST SUITE COMPLETE")
        print(f"   Total tests: {total_tests}")
        print(f"   Failed tests: {failed_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Corruption events: {len(self.corruption_events)}")
        print(f"   Total time: {elapsed_time:.1f}s")
        print(f"   Average time per test: {elapsed_time / total_tests:.2f}s")

        return {
            "total_tests": total_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "corruption_events": len(self.corruption_events),
            "elapsed_time": elapsed_time,
        }

    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous background monitoring"""
        if self.is_monitoring:
            print("❌ Monitoring already running!")
            return

        print(f"🔄 Starting continuous monitoring (interval: {interval}s)")
        self.is_monitoring = True

        def monitor_loop():
            cycle = 0
            while self.is_monitoring:
                cycle += 1
                # Run a random test command
                test_cmd = self.test_commands[cycle % len(self.test_commands)]

                try:
                    self.run_single_test(test_cmd)
                except Exception as e:
                    self.log("ERROR", f"Monitoring cycle {cycle} failed: {e}")

                time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_monitoring:
            print("❌ Monitoring not running!")
            return

        print("🛑 Stopping continuous monitoring...")
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("✅ Monitoring stopped")

    def generate_report(self) -> dict:
        """Generate comprehensive monitoring report"""
        report = {
            "summary": {
                "total_tests": self.test_counter,
                "corruption_events": len(self.corruption_events),
                "corruption_rate": len(self.corruption_events) / max(1, self.test_counter),
                "monitoring_started": datetime.datetime.now().isoformat(),
            },
            "corruption_patterns": {},
            "stage_analysis": {},
            "command_analysis": {},
        }

        if self.corruption_events:
            # Analyze corruption by type
            corruption_types = {}
            stages = {}
            commands = {}

            for event in self.corruption_events:
                # By type
                corruption_types[event.corruption_type] = corruption_types.get(event.corruption_type, 0) + 1

                # By stage
                stages[event.stage] = stages.get(event.stage, 0) + 1

                # By command pattern
                cmd_pattern = event.command_sent[:20] + "..." if len(event.command_sent) > 20 else event.command_sent
                commands[cmd_pattern] = commands.get(cmd_pattern, 0) + 1

            report["corruption_patterns"] = corruption_types
            report["stage_analysis"] = stages
            report["command_analysis"] = commands

        return report

    def print_status(self):
        """Print current monitoring status"""
        print("\n📋 TERMINAL MONITOR STATUS")
        print(f"   Tests run: {self.test_counter}")
        print(f"   Corruption events: {len(self.corruption_events)}")
        print(f"   Monitoring active: {self.is_monitoring}")
        print(f"   Log file: {self.log_file}")

        if self.corruption_events:
            print("\n🚨 RECENT CORRUPTION EVENTS:")
            for event in self.corruption_events[-3:]:  # Show last 3
                print(f"   {event.timestamp}: {event.corruption_type}")
                print(f"      '{event.command_sent}' -> '{event.command_received}'")


# MAIN FUNCTION FOR EASY USAGE
def monitor_terminal(terminal_instance, mode="test", **kwargs):
    """
    Easy interface to monitor terminal corruption

    Modes:
    - 'test': Run test suite once
    - 'continuous': Start continuous monitoring
    - 'stress': Run stress test with many iterations
    """
    monitor = TerminalCorruptionMonitor(terminal_instance)

    try:
        if mode == "test":
            iterations = kwargs.get("iterations", 1)
            return monitor.run_test_suite(iterations)

        elif mode == "continuous":
            interval = kwargs.get("interval", 1.0)
            monitor.start_continuous_monitoring(interval)
            print("Press Ctrl+C to stop monitoring...")
            try:
                while monitor.is_monitoring:
                    time.sleep(1)
                    if hasattr(monitor, "_should_print_status"):
                        monitor.print_status()
            except KeyboardInterrupt:
                monitor.stop_continuous_monitoring()

        elif mode == "stress":
            iterations = kwargs.get("iterations", 10)
            return monitor.run_test_suite(iterations)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    finally:
        # Always generate final report
        report = monitor.generate_report()
        report_file = monitor.log_dir / f"final_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Final report saved to: {report_file}")
        monitor.print_status()

        return monitor


# USAGE EXAMPLES:
# Basic usage - run once
monitor = monitor_terminal(your_terminal_instance, mode="test")

# Stress test - many iterations
monitor = monitor_terminal(your_terminal_instance, mode="stress", iterations=50)

# Continuous monitoring
monitor = monitor_terminal(your_terminal_instance, mode="continuous", interval=0.5)

# Manual control
monitor = TerminalCorruptionMonitor(your_terminal_instance)
monitor.run_test_suite(5)
monitor.start_continuous_monitoring(1.0)
# ... do other work ...
monitor.stop_continuous_monitoring()
print(monitor.generate_report())
