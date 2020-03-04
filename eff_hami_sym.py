import sympy as sy

delta_l = sy.Symbol(r'\Delta_l')
delta_r = sy.Symbol(r'\Delta_r')
q_0l = sy.Symbol(r'q_{0l}')
q_0r = sy.Symbol(r'q_{0r}')
Nl = sy.Symbol(r'N_l')
Nr = sy.Symbol(r'N_r')
H_eff = sy.Matrix([[0, delta_l*sy.E**(Nl*q_0l), 0, 0],[delta_l*sy.E**(-Nl*q_0l), 0, 0, 0],[0, 0, 0, delta_r*sy.E**(Nr*q_0r)],[0, 0, delta_r*sy.E**(-Nr*q_0r),0]])
print(sy.latex(H_eff.diagonalize()))
