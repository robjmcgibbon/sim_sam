# Converting simulations into SAMs

### Current goals

- Extract efficiencies from TNG. Generate SAM galaxies using the coefficients. Compare individual galaxies, and the population as a whole.
- Determine an equation for the efficiencies using symbolic regression
- Repeat for other TNG resolutions, original Illustris data

### Extensions

- Get data from Mitchell paper (he has left academia)
- Calculate outflows for TNG
- Include accreted stellar mass

### Meeting - 6/7/22

- I am using set operations to calculate the rates. I could use a Venn diagram if I want a visual representation of how the rates are calculated.
- Early results (and results from Neistein paper) show that $f_c$ decreases with redshift. This is probably because of Figure 5 in [this paper](https://arxiv.org/abs/0808.0553) about cold mode accretion. At high redshift cold mode accretion occurs which gets cold gas into halos. Also at higher redshift the halos have lower mass on average, so the virial temperature is lower.

### Technical notes
- There is [an issue](https://github.com/conda/conda/issues/11580) with pip freeze
- Pandas > 1.2 does not work on the fcfs nodes (Illegal Instruction)


