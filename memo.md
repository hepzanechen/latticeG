To make all things get done in GPU, E should be inited at GPU, the device of E it's the standard intermeidate tesor's device choose to follow.
Also lead's properties, central's properties should be checked.
the device is contrled by funcDevice, set to Ebatch device or lead.v1laph device in ginv_total

* lead_decimation calc hole and electron seperately and combine them in add_ginv_lead, and inv the whole, this may be improved by 1. pass in H_lead_BdG together, 2. use the hole and electron seperately to calc the hole and electron seperately, and combine them in add_ginv_lead use symmetric property.
* tLC can only be real number for now
* since $$ is current, the hermitial part automatically handles the minius sign needed becuae $\exp{(-i \lambda)}$ also reverse sign due to hermitian.

$ \ln S \propto \frac{1}{2}(\psi_{1}^{\dagger}(E)\psi_{2}(E) + \psi_{2}^{\dagger}(E)\psi_{1}(E)) - \frac{1}{2}(\psi_{2}(-E)\psi_{1}^{\dagger}(-E) + \psi_{1}(-E)\psi_{2}^{\dagger}(-E)) $

$-E$ means $\omega\psi_{1}^{\dagger}\psi_{1} \rightarrow -\omega\psi_{1}\psi_{1}^{\dagger}$ corresponding hole energy is $-E$ and this is due to p-h symmetry redundancy, while $-\frac{1}{2}(\psi_{2}(-E)\psi_{1}^{\dagger}(-E))$ means $t\psi_{1}^{\dagger}\psi_{2} \rightarrow -t\psi_{2}\psi_{1}^{\dagger}$

It should be $tLch = -tLce^{\dagger}$

* eta can causing to lead current conservation fails, so we should let eta=0

* since electron and hole are $E<->-E$ symmetry, so there are issues if $\mu_1=-\mu_2$, then physically there should be curren, but the calculated current is not zero, so we should make hole part $-E$ in the gen_ginv_lead.
