# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parameter object used within calibration framework."""

import hashlib
from typing import Tuple

from qiskit.circuit.parameterexpression import ParameterExpression
from sympy import Symbol


class CalibrationParameter(ParameterExpression):

    def __init__(self,
                 name: str,
                 instruction: str = None,
                 pulse: str = None,
                 qubits: Tuple[int] = None):
        """Create new parameter object.

        Args:
            name: Name of this parameter, e.g. `amp`, `duration`, ...
            instruction: Gate instruction that this parameter is map to.
            pulse: Pulse name that this parameter is bound to.
            qubits: Target qubits the instruction applied to.
        """
        if qubits is not None:
            try:
                qubits = tuple(qubits)
            except TypeError:
                qubits = (qubits, )
        else:
            qubits = tuple()

        self._instruction = instruction or ''
        self._pulse = pulse or ''
        self._qubits = qubits

        base_str = f'{instruction}.{pulse}.{".".join(map(str, qubits))}'
        self._scope = hashlib.md5(base_str.encode('utf-8')).hexdigest()

        self._name = name

        symbol = Symbol(name)
        super().__init__(symbol_map={self: symbol}, expr=symbol)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        """Return the scope of this parameter."""
        return self._scope

    @scope.setter
    def scope(self, new_scope: str):
        """Overwrite scope."""
        self._scope = new_scope

    def to_dict(self):
        return {
            'instruction': self._instruction,
            'pulse': self._pulse,
            'qubits': self._qubits,
            'name': self.name,
            'scope': self.scope
        }

    def subs(self, parameter_map: dict):
        return parameter_map[self]

    def __hash__(self):
        return hash((self.name, self.scope))

    def __eq__(self, other):
        if isinstance(other, CalibrationParameter):
            return self.name == other.name and self.scope == other.scope
        elif isinstance(other, ParameterExpression):
            return super().__eq__(other)
        else:
            return False
