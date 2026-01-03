from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Union

Number = Union[int, float]

class Value(object):
    def __init__(self, data: Number, _children: Tuple[Value, ...] = (), _op: str = '', label: str = '') -> None:
        self.data: float = float(data)
        self._prev: set[Value] = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
    
    def __neg__(self) -> Value:
        out = Value(-self.data, (self,), 'neg')

        def _backward():
            self.grad += -out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other: Value | Number) -> Value:
        return self + (-other)
    
    def __rsub__(self, other: Value | Number) -> Value:
        return other + (-self)

    def __add__(self, other: Value | Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other: Value | Number) -> Value:
        return self + other

    def __mul__(self, other: Value | Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other: Value | Number) -> Value:
        return self * other
    
    def __truediv__(self, other: Value | Number) -> Value:
        return self * other**-1
    
    def __rtruediv__(self, other: Value | Number) -> Value:
        return other * self**-1

    def __pow__(self, exponent: Number) -> Value:
        n = self.data
        out = Value(n**exponent, (self,), f'pow_{exponent}')

        def _backward():
            self.grad += (exponent * n**(exponent - 1)) * out.grad
        out._backward = _backward

        return out
    
    def exp(self) -> Value:
        n = self.data
        out = Value(np.exp(n), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def tanh(self) -> Value:
        e = (2*self).exp()
        t = (e - 1) / (e + 1)
        out = Value(t.data, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t.data**2) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
