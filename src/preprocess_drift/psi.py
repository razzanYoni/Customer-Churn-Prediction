def isRetrain(psi_values) -> bool:
    return any(psi > 0.1 for psi in psi_values.values())

