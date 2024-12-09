To make all things get done in GPU, E should be inited at GPU, the device of E it's the standard intermeidate tesor's device choose to follow.
Also lead's properties, central's properties should be checked.
the device is contrled by funcDevice, set to Ebatch device or lead.v1laph device in ginv_total

* lead_decimation calc hole and electron seperately and combine them in add_ginv_lead, and inv the whole, this may be improved by 1. pass in H_lead_BdG together, 2. use the hole and electron seperately to calc the hole and electron seperately, and combine them in add_ginv_lead use symmetric property.
* tLC can only be real number for now