import libcst

from swesmith.bug_gen.procedural import BaseProceduralModifier
from swesmith.bug_gen.criteria import *


class RemoveLoopModifier(BaseProceduralModifier):
    explanation: str = "There is one or more missing loops that is causing the bug."
    name: str = "func_pm_remove_loop"
    conditions: list = [
        filter_functions,
        filter_loops,
        filter_min_simple_complexity,
    ]  # Assuming filter functions will be applied externally

    def leave_For(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node

    def leave_While(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node


class RemoveConditionalModifier(BaseProceduralModifier):
    explanation: str = "There is one or more missing conditionals that causes the bug."
    name: str = "func_pm_remove_cond"
    conditions: list = [
        filter_functions,
        filter_conditionals,
        filter_min_simple_complexity,
    ]

    def leave_If(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node


class RemoveAssignModifier(BaseProceduralModifier):
    explanation: str = "There is likely a missing assignment in the code."
    name: str = "func_pm_remove_assign"
    conditions: list = [
        filter_functions,
        filter_assignments,
        filter_min_simple_complexity,
    ]

    def leave_Assign(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node

    def leave_AugAssign(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node


class RemoveWrapperModifier(BaseProceduralModifier):
    explanation: str = "There are missing wrappers (with, try blocks) in the code."
    name: str = "func_pm_remove_wrapper"
    conditions: list = [filter_functions, filter_wrappers, filter_min_simple_complexity]

    def leave_With(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node

    def leave_AsyncWith(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node

    def leave_Try(self, original_node, updated_node):
        return libcst.RemoveFromParent() if self.flip() else updated_node
