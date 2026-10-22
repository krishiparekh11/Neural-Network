"""Demo: McCulloch-Pitts neuron simulation.

Run:  python main.py
"""
from itertools import product

from mcp_neuron import MCPNeuron
from mcp_neuron.logic_gates import AND, OR, not_gate, xor_gate


def truth_table(title, fn, n_inputs):
    print(f"\n{title}")
    print("  " + "  ".join(f"x{i+1}" for i in range(n_inputs)) + " | y")
    for combo in product([0, 1], repeat=n_inputs):
        print("  " + "   ".join(str(v) for v in combo) + "  | " + str(fn(*combo)))


def interactive():
    print("\n--- Try your own neuron ---")
    try:
        n = int(input("Number of inputs N: "))
        theta = float(input("Threshold theta: "))
        neuron = MCPNeuron(n, theta)
        raw = input(f"Enter {n} Boolean inputs separated by spaces: ").split()
        x = [int(v) for v in raw]
        print(f"g = {neuron.aggregate(x)}, y = {neuron.fire(x)}")
    except (ValueError, EOFError) as err:
        print(f"Skipped: {err}")


if __name__ == "__main__":
    and_n, or_n = AND(), OR()
    print(and_n)
    truth_table("AND (theta = 2)", lambda a, b: and_n.fire([a, b]), 2)
    truth_table("OR (theta = 1)", lambda a, b: or_n.fire([a, b]), 2)
    truth_table("NOT (theta = 0, inhibitory input)", not_gate, 1)
    truth_table("XOR = AND(NOT(AND), OR)", xor_gate, 2)

    and3 = AND(3)
    truth_table("3-input AND (theta = 3)", lambda a, b, c: and3.fire([a, b, c]), 3)

    majority = MCPNeuron(3, theta=2, name="MAJORITY")
    truth_table("3-input MAJORITY (theta = 2)",
                lambda a, b, c: majority.fire([a, b, c]), 3)

    interactive()
