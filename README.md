# Converting simulations into SAMs

### Current goals
- Extract efficiencies from TNG. Generate SAM galaxies using the coefficients. Compare individual galaxies, and the population as a whole.
- Determine an equation for the efficiencies using symbolic regression
- Repeat for other TNG resolutions, original Illustris data

### Extensions
- Implement loss of stellar mass due to stellar evolution
- Get data from Mitchell paper (he has left academia)
- Calculate outflows for TNG
- Separate hot and cold mode accretion
- Include negative accretion in mode (Neistein sec 3.1)
- Could switch to a one phase model (Neistein sec 5)
- Could assume a value for f_c to make f_d easier to calculate

### Problems
- What happens when cells split?
- Mass flows into and out of cells

### Technical notes
- There is [an issue with pip freeze](https://github.com/conda/conda/issues/11580)
- Pandas > 1.2 does not work on the fcfs nodes (Illegal Instruction)
- Julia is a requirement for PySR

# Notes
### gplearn implementation
Fitness evaluation is defined in _program.py \
Fitness is defined as raw_fitness - (parsimony_coefficient * len(program))

# Meeting notes

### 27/07/19
- I have two efficiencies for accrete rate of hot gas.
One is from the particle ids (f_m_id) and significantly overestimates the hot gas mass in a halo.
I think this is because whenever a cell splits this method counts it as being accreted.
The second method (f_m) is to use the difference in baryonic mass, assuming that all gas that accretes goes to hot gas.
- The second method still gives a number of halos with high accretion efficiencies, but we think these are due to numeric effects.

### 19/07/22
- Value of $f_c$ seems consitent for me and N12. This is because $f_c$ depends on the cooling time, which depends on the halo mass. Therefore it's a function of cosmology more than subgrid models.
- $f_s$ is also going to be similar for simulations as a KS law is universally used for star formation in simulations.
- Mass function of centrals vs satellites shows why it's valid to only consider centrals as long as there is a high enough mass cut.
- The baryon fraction of halos has an interesting shape. The general increase with mass is due to an increased ability to hold on to the gas. We think the drop is due to BH feedback.

### 6/07/22
- I am using set operations to calculate the rates. I could use a Venn diagram if I want a visual representation of how the rates are calculated.
- Early results (and results from Neistein paper) show that $f_c$ decreases with redshift. This is probably because of Figure 5 in [this paper](https://arxiv.org/abs/0808.0553) about cold mode accretion. At high redshift cold mode accretion occurs which gets cold gas into halos. Also at higher redshift the halos have lower mass on average, so the virial temperature is lower.



