---
layout: post
title: "AI is turning nuclear: a review"
tags: [AI, Energy, Climate]
comments: false
---

## Will nuclear power satiate AI energy hunger?

![_config.yml]({{ site.baseurl }}/images/nuclear_ai.png)

_This image was generated using FLUX1-dev_

### AI, data and energy: an introduction

November 2022 changed the life of humans forever: the world of Artificial Intelligence, that had been operating for years out of the spotlight, finally came to the limelights with OpenAI's ChatGPT, a chat interface that leveraged a Large Language Model (GPT-3) to generate responses to the humans it interacted with.
The excitement around AI exited then for the first time the scientific community, reaching also the business world: in almost two years, investments and revenues in the field rocketed, with big and small companies pushing the revolution further, testing the limits of our technologies.

In less than two years, from GPT-3 to Llama-3, the data volumes for AI went up from 10^11 to 10^13 training tokens, and this data hunger, combined with the need for computational power, will drive the increase in data centers' energy demand to almost double its current size in 2030. 

Environmental costs of Artificial Intelligence are pretty much obscure, due to non-disclosure policies of the companies that build the most of it, but the path is clear: its power needs will be huge, and the consequences on the electrical consumption will be very relevant.

The question now is: how will we be able to power this revolution without worsening the already dramatic climate crisis we're going through?

### Understanding the problem: some key facts

**1. AI companies are investing in more powerful hardwares**

Following [Beth Kindig's steps on Forbes](https://www.forbes.com/sites/bethkindig/2024/06/20/ai-power-consumption-rapidly-becoming-mission-critical/), we can see that hardware-producing companies, such as NVIDIA, AMD and Intel, are putting money into more and more powerful chips, able to manage larger data volumes in a fast and efficient way, but with increased power requirements: 

- Up to now, the two most powerful NVIDIA GPU hardwares, A100 and H100, consume respectively 250W/chip and 300 to 700W/chip when brought to the maximum power. The next generation GPUs, Blackwell's series B200 and GB200, will be able to run at 1200 and 2700W/chip, with a 4-fold increase in their power consumption 
- AMD's most powerful GPU hardware, MI300x, consumes 750W/chip, up to 50% compared to its predecessor MI250
- Intel is currently working on the _Falcon shores_ chips, which will have a 1500W/chip power consumption, a 67% increase if compared to Gaudi 3, which "only" consumes 900W.

**2. AI developers are pushing to build bigger powerhouses for their models**

Training and running models takes a huge toll of computation and data flow, which, with the scaling up of AI revolution, will become bigger every year, requiring larger and larger physical infrastructures where to fuel this computational power:

- In summer 2024, xAI announced through Elon Musk that they built a 100.000 H200 GPUs powerhouse where to run and train the latest versions of their model Grok
- Meta, in their [_Building Meta's GenAI infrastructure_](https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/) statement, announced that it will focus its investments on two 24.000 GPU clusters, and said that: "By the end of 2024, we’re aiming to continue to grow our infrastructure build-out that will include 350,000 NVIDIA H100 GPUs as part of a portfolio that will feature compute power equivalent to nearly 600,000 H100s.".
- Google announced that it is investing $3 billion dollars in South Eastern Asia, especially Malaysia and Thailand, to expand its AI capabilities and cloud infrastructure

**3. AI is not as green as we think**

AI already huge power consumption is estimated to grow 10 times by 2026, [surpassing the power requirements of a small country like Belgium](https://hbr.org/2024/07/the-uneven-distribution-of-ais-environmental-impacts). This demand does not come without a cost: despite claims of "greenness" by companies, the impact on the environment is way more complex than it appears, and it goes beyond the emissions:

- In 2022, Google claimed that its data center in Finland run on 98% carbon-free energy. This percentage, nevertheless, goes down to 4-18% in Asian data centers, exactly where Google is now pouring money to build new infrastructure.
- In 2019, [Microsoft announced their partnering with ExxonMobil](https://news.microsoft.com/2019/02/22/exxonmobil-to-increase-permian-profitability-through-digital-partnership-with-microsoft/), one of the biggest oil companies in the world: thanks to several AI tools, ExxonMobil announced they optimized oil extraction and will be able to increase it by 50.000 barrels/day in 2025
- According to a [2023 research study](https://arxiv.org/abs/2304.03271), AI is not only hungry for energy, it is also thirsty for water: water is one of the most used coolers for data centers, which makes it crucial to maintain them at an optimal performance status. This is even more important in hot areas like Arizona data centers, where temperatures reach high peaks during summer and water becomes scarce. The estimated water volumes needed by AI _per se_ in 2027 are 4.2 to 6.6 billion cubic meters, like the water consumption of the entire UK, and training GPT-3 alone in Microsoft SOTA data centers required 700.000 liters of fresh water.
- In its 2024 environmental report, Google claimed that AI-driven energy requirements in data centers brought [their greenhouse gases emissions up by 48%](https://www.bbc.com/news/articles/c51yvz51k2xo)

Summing everything up, AI is **growing fast**, hardware producers are making it **more and more power demanding**, big tech companies are pouring billions into **huge computational and data factories** to cope with the growth of the sector, and the resulting **impact** on the environment, both **direct and indirect**, is becoming more and more **relevant**. 

### Going nuclear: the solution?

**1. The context**

Although not as concerned as environmental scientists are, big tech companies are still driven by money and practicality: if the energy requirements of AI become too big and they are not able to provide enough electricity to satisfy them, the game will be over for everyone.

In this sense, [Microsoft](https://www.theguardian.com/environment/2024/sep/20/three-mile-island-nuclear-plant-reopen-microsoft), [Amazon](https://www.aboutamazon.com/news/sustainability/amazon-nuclear-small-modular-reactor-net-carbon-zero) and [Google](https://blog.google/outreach-initiatives/sustainability/google-kairos-power-nuclear-energy-agreement/) announced that they will all be involved in some nuclear-related project, renting, acquiring or building from scratch new nuclear-fuelled power plants to help with the energy demand:

- **Microsoft** will restart Three Miles nuclear power plant in Pennsylvania, home to the biggest nuclear leak in the USA history, to generate **835 megawatts** (MW) of energy to put in their grid.
- **Amazon** will rely on the public consortium Energy Northwest to build four Small Modular Reactors to reach a total power of **960 MW** at full capacity, an equivalent of the power consumed by 770.000 American households.
- **Google** partnered with Kairos Power to deploy several Small Modular Reactors to bring online by 2030 and some others by 2035, for a total of **500 MW** of power

To understand the importance of these decisions, we have to understand why nuclear is being chosen over other technologies and what are the Small Modular Reactors on which the big techs are betting.

**2. Nuclear energy**

The debate on nuclear energy has been going on for decades, and concerned its safety, its impact on the environment and the consequences on human and animal health. To understand its importance beyond political and ideological factions, let's get some facts straight:

- Nuclear energy is produced via nuclear fission, a process that involves bombarding the nucleus of unstable radioactive elements (like uranium) with neutrons: this activates a cascade of events which, in a controlled environment, frees usable energy that comes from the stabilization of the atomic nuclei. This happens because, generally, a radioactive nucleus loses energy going from an unstable to a stable form, energy which can be piped into stable channels and served to an electrical grid.
- Nuclear energy does not require anything to be burnt, does not involve greenhouse gases emissions and yields high amounts of energy with relatively low quantity of radioactive material: natural uranium in a fast-breeder reactor has an energy density of approx. [86 million joules per kilogram, 3.6 million times higher than coal](https://en.m.wikipedia.org/wiki/Energy_density_Extended_Reference_Table)
- There are now [440 reactors distributed in 31 countries all around the world](https://world-nuclear.org/information-library/current-and-future-generation/nuclear-power-in-the-world-today) that, in 2023, satisfied 10% of the global electricity demand
- Safety concerns about potential nuclear incidents due to bad constructions are well behind us, being the current safety protocols very meticulous and solid. Nevertheless, we still have the problem of ['nuclear waste'](https://world-nuclear.org/nuclear-essentials/what-is-nuclear-waste-and-what-do-we-do-with-it), which is composed by all exhausted radioactive or radiation-exposed materials. Although not being a main concern now, nuclear waste has to be disposed: as of today, the simplest solution would be to put it underground, in caves where it would stay far apart from humanity for hundreds of thousands of years.
- The main problem to implement nuclear energy on a large scale are [the surging costs](https://www.energypolicy.columbia.edu/the-uncertain-costs-of-new-nuclear-reactors-what-study-estimates-reveal-about-the-potential-for-nuclear-in-a-decarbonizing-world/) (that in the USA range approx. from 3000 to 6000 $/kWh) that are required to build reactors and the not-so-quick construction times ([average is 11-12 years](https://www.statista.com/statistics/712841/median-construction-time-for-reactors-since-1981/), with relevant exceptions)

So nuclear energy, although **not being renewable** (it depends on radioactive materials, which are a limited resource), is green and strongly effective, but suffers from high production costs and long construction times, apart from the problem of nuclear waste.

**3. Small Modular Reactors**

One potential solution to the problems that affect nuclear energy development are [Small Modular Reactors](https://www.energy.gov/ne/benefits-small-modular-reactors-smrs) (SMR) which are, as the name suggests, smaller implementations of the traditional power plants. 

- They are small and modular, so their modules can be pre-assembled in a factory and just combined into a reactors _in loco_, speeding up significantly the construction times and dramatically cutting the costs.
- Their security is managed without complex systems: being small and not dealing with high quantities of energy, these reactors take advantage of naturally-occurring physical processes to safeguard the energy production
- They have a good energy efficiency: even though they produce a third of the energy that generally a traditional reactor outputs, they can be coupled with renewable sources of energy to enhance their performances.

Despite the obvious advantages, lots SMRs are still in the designing phase, and there is not enough evidence to assess their nuclear waste production: [a research by Standford and British Columbia University](https://news.stanford.edu/stories/2022/05/small-modular-reactors-produce-high-levels-nuclear-waste) suggests indeed that they would produce (in proportion) more waste than traditional reactors, compared to an energy production which still does not surpass the 300 MW/reactor. 

So this leads to our big question, but also conclusion: 

**4. Why are Big Tech turning nuclear for AI?**

As we saw, nuclear energy is highly efficient and, with technological advancements such as SMRs, is becoming more and more feasible and scalable. Apart from the nuclear waste problem (which can still constitute a big issue on the long run), nuclear energy is clean and carbon-free, so it does not contribute to the climate crisis. All of these reasons make it the perfect candidate to "clean" AI while yielding more power for it, even though some key points still remain unclear:

- Big techs are pushing to build nuclear power but their energy requirements are way larger than what could be provided by those SMRs only: Google alone, according to its own environmental report, consumed 24 TWh of electricity in 2024, which means 24 millions MWh. The SMRs could contribute for a very small part, which probably will be piped straight into GenAI data centers and facilities, but they alone won't actually be able to satisfy the ever growing energy hunger of AI. 
- These projects, even though planned on the short term (most of them will be carried out before 2035-2040), will take time, but the AI boom is happening now and the surging demand will be a problem way before 2035-2040: what will the strategy of the big techs be for the time being?
- Besides investments in nuclear energy, big techs will need to give their money also to clean energy facilities. What they've been doing up to now, tho, has been acquiring Renewable Energy Credits (RECs) as a workaround: arguing that getting an entirely clean and green stream of renewable energy is almost impossible, tech giants just give money to developers that ensure that they'll use those investments to build new renewable energy infrastructures. Another widely used model are carbon credits (CCs), a financial instrument that allows a company to pay someone else to take action and reduce their carbon emissions. RECs and CCs combined are a cheap and easy way to claim environmental goals without actually having met them in practice: according to a [review by MIT](https://www.technologyreview.com/2024/07/17/1095019/google-amazon-and-the-problem-with-big-techs-climate-claims/), this strategy is widely used (Google, Amazon, Meta and Salesforce are just some examples) and often brings to no/scarce actual results in lowering a company's impact, despite the claims of carbon neutrality.
- Electrical grids are becoming every day more stressed because of the needs for energy by data centers and computational facilities: how will they handle the incoming power that is being poured into them to feed the demand of AI?

So, in conclusion: are big techs really interested in the decarbonizing potential of nuclear energy, apart from its power efficiency, or are they just energy-hungry and trying to find some short-term cost effective solutions which will also allow them to green-wash their image? There is no easy answer, and maybe there is no answer at all, for now: only the future will tell us what side they took.

### References

See the references for this article [here](https://raw.githubusercontent.com/AstraBert/hophop-science/main/images/references_2024_20_10.txt
)