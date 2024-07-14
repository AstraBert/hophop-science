---
layout: post
title: "Is AI carbon footprint worrisome?"
tags: [AI, Energy, Climate]
comments: false
---

>_"AI-powered robots are deployed by the farming industry to grow plants using less resources (Sheikh, 2020). AI is applied to tackle the protein structure prediction challenge, which can lead to revolutionary advances for biological sciences (Jumper et al., 2021). AI is also used to discover new electrocatalysts for efficient and scalable ways to store and use renewable energy (Zitnick et al., 2020) while also being applied to predict renewable energy availability in advance to improve energy utilization (Elkin & Witherspoon, 2019)"_ - From Wu et al., 2022

![_config.yml]({{ site.baseurl }}/images/is_ai_cf_worrisome.png)

The juxtaposition (and contraposition) of the two sets of statements at the beginning of this article does not come without a precise intention: it wants to underline one of the biggest contrasts of AI, a paradox-like loophole in which a tool that can help us through the climate crisis may, in the future, be an active part of that same crisis. 

Is AI really that environmentally-threatening? Is there anything we could do to improve this situation? Let's break this down, one step at a time.

## 0. Before we start: a little bit of terminology

We need to introduce three main terms, that we'll be using throughout the article and that will be a useful common ground to agree on:

- Carbon footprint: according to [Britannica](https://www.britannica.com/science/carbon-footprint), it is the "amount of carbon dioxide (CO2) emissions associated with all the activities of a person or other entity (e.g., building, corporation, country, etc.)". This does not only mean how much fossil fuels one _directly_ consumes (gasoline, plastics...), but also all the emissions necessary for transportation, heating, electricity in the process of production of goods and provision of services.
- CO2e (equivalent CO2): the [European Commission](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Carbon_dioxide_equivalent) writes that it is "a metric measure used to compare the emissions from various greenhouse gases on the basis of their global-warming potential (GWP), by converting amounts of other gases to the equivalent amount of carbon dioxide with the same global warming potential". This simply means that there are lots of other greenhouse gases (methane, clorofluorocarbons, nitric oxide...) which all have global warming potential: despite our emissions being mainly made up by CO2, they encompass also these other gases, and it is easier for us to express everything in terms of CO2. For example: methane has 25 times higher global warming power than CO2, which means that producing 1 kg of methane can be translated into producing 25 kg of CO2e.
- Life cycle assessment (LCA): following [European Environmental Agency](https://www.eea.europa.eu/help/glossary/eea-glossary/life-cycle-assessment) glossary, LCA "is a process of evaluating the effects that a product has on the environment over the entire period of its life thereby increasing resource-use efficiency and decreasing liabilities". We can use this technique to trace the impact of an object (or sometimes a service) from start to end, understanding the energetic consumptions associated with its production, use and disposal.

These three definitions come with a disclaimer (especially for the first and last one): not everybody in the scientific community believes they are correct, and there are several other possibilities to define these concepts. What is interesting to us in this article is to grasp an operative knowledge, that will allow the understanding of facts and figures about AI impact on the environment: we won't, thus, dive into scientific terminological disputes.  

## 1. AI impact on the environment: a troubled story

There is a great problem about AI carbon footprint: we know very little about it, and most of AI companies are not really transparent on those data.

Let's, nevertheless, try to look at some estimates, following a paper (_Sustainable AI: Environmental Implications, Challenges And Opportunities_) coming out of the 5th MLSys Conference, held in Santa Clara in 2022. The main idea behind the proposed analysis is to follow AI consumptions end-to-end, from hardware production to usage to deployment, in what the authors define as a "holistic approach":

- _Hardware production, usage, maintenance and recycling_: this portion is based on a thorough LCA for processors and other hardware facilities: the conclusion seems to point to a 30/70% split between hardware (or embodied) and computational (or operational) carbon footprint. 
- _Researching, experimenting and training_: although researching and experimenting could take long times and relevant computational efforts, these two portions are not nearly as heavy as training in terms of carbon footprint. A model like GPT-3, which we deem as surpassed nowadays, required >600.000 kg of CO2e: considered that the average world carbon footprint per person is about 4000 kg/year, we can say that GPT-3 had as much impact as 150 people in one year. Moreover, you have to consider that there is not only "offline" training (the one done with historical data), but there's also "online" training, the one that keeps models up-to-date with recently published content: this portion, for example, is particularly relevant to recommendations models such as Meta's RM1-5.
- _Inference_: Inference may be the most relevant portion in terms of carbon costs: as Philip Lewer (Untether AI) says, "models are built expressly for the purpose of inference, and thus run substantially more frequently in inference mode than training mode — in essence train once, run everywhere" (from [this article](https://semiengineering.com/ai-power-consumption-exploding/)). According to researchers from MIT and Northeastern University, "different estimates from NVIDIA and Amazon suggest that inference tasks account for 80% or more of AI computational demand" (MacDonald et al., 2022). Also a model like RM1 at Meta, inference almost doubles the carbon costs already produced by offline and online training.

## 2. Data craving: an energy problem

If all of these aspects account for a relevant portion of AI carbon footprint, there's also another giant elephant the room that we've been ignoring up to this point: data. While not directly linked to AI "hardware" lifecycle, they are a crucial part for building models: data volumes in the LLM field went from an order of 10^11 tokens for GPT-3 (2020-21) to surpassing 10^13 tokens for Llama 3. [Epoch AI's estimates](https://epochai.org/blog/will-we-run-out-of-data-limits-of-llm-scaling-based-on-human-generated-data) tell us that we're going to run out of human-generated data to train AI between 2026 and 2032.

Where do we put and how do we maintain all these data? Data centers consumed 460 TWh of electric energy in 2022, accounting for 2% of World's demand: [according to the International Energy Agency](https://www.datacenterfrontier.com/energy/article/33038469/iea-study-sees-ai-cryptocurrency-doubling-data-center-energy-consumption-by-2026), data centers have the potential to double their consumes by 2026, with AI and cryptocurrencies leading the increase. 

But why do data centers require so much energy? This is not only to keep their supercomputers going 24/7, but it is prominently to avoid overheating: a good share of the energy is indeed absorbed by cooling systems (and this may not be only an electricity problem, but also [a water one](https://www.asce.org/publications-and-news/civil-engineering-source/civil-engineering-magazine/issues/magazine-issue/article/2024/03/engineers-often-need-a-lot-of-water-to-keep-data-centers-cool)). As underlined by MacDonald et al. in their paper, energy expenses are temperature-sensitive, which means that, with global warming, cooling may require even more effort.


## 3. Can we do something? An outlook

Researchers have been exploring numerous solutions to the problem of AI carbon footprint: [Google](https://research.google/blog/good-news-about-the-carbon-footprint-of-machine-learning-training/), for example, in 2022 proposed the 4Ms to reduce the carbon footprint of Machine Learning and Deep Learning:

- _Model_: optimize model choice, preferring sparse over dense models, as they require less computational energy (3x to 10x reduction)
- _Machine_: use specifically tailored hardwares (like TPUv4) to reduce losses and increase efficiency (2x to 5x optimization).
- _Mechanization_: computing in the cloud and using cloud data centers instead of physical ones can contribute to the decrease of energy consumptions by 1.4x to 2x
- _Map optimization_: choosing the right location to sustain your cloud can significantly improve your carbon footprint reduction contributing with another 5x to 10x.

Also LMSys 2022 paper highlighted a combination of techniques that they used to reach an overall 810x energy consumption reduction in relation to Meta CPU carbon costs baseline:

- _Platform-level caching_: frequently accessed data and embedding are precomputed and cached inside a DRAM which makes them accessible in an easier way.
- _GPU usage_: employing GPU acceleration can decrease energy up to 10x
- _Low precision data format_: employing FP16 GPUs instead of FP32 ones proved more efficient
- _Algorithm optimization_: choosing the right training and inference algorithms can decrease energy costs up to 5x

Still, questions remain: will all these procedures really help us decrease AI impact on the environment? Will AI itself prove more beneficial for climate crisis that it will be detrimental? 

Beyond these questions and all the others that may be asked, what stands out clear from all these observations is that, beyond questioning, we need to start taking action, requesting transparency and green policies from AI companies and starting building climate-awareness around our own AI use. And then, at the right time, the answers will come.

## References

- Wu, C. J., Raghavendra, R., Gupta, U., Acun, B., Ardalani, N., Maeng, K., ... & Hazelwood, K. (2022). Sustainable ai: Environmental implications, challenges and opportunities. Proceedings of Machine Learning and Systems, 4, 795-813.
- McDonald, J., Li, B., Frey, N., Tiwari, D., Gadepally, V., & Samsi, S. (2022). Great power, great responsibility: Recommendations for reducing energy for training language models. arXiv preprint arXiv:2205.09646.
- Cho R. (2023) AI growing carbon footprint, https://news.climate.columbia.edu/2023/06/09/ais-growing-carbon-footprint/
- De Bolle M. (2024) AI's carbon footprint appears likely to be alarming, https://www.piie.com/blogs/realtime-economics/2024/ais-carbon-footprint-appears-likely-be-alarming
- Bainley B. (2022) AI Power Consumption Exploding, https://semiengineering.com/ai-power-consumption-exploding/
- Heikkilä M. (2023) AI's carbon footprint is bigger than you think https://www.technologyreview.com/2023/12/05/1084417/ais-carbon-footprint-is-bigger-than-you-think/
- Patterson D. (2022) Good News About the Carbon Footprint of Machine Learning Training, https://research.google/blog/good-news-about-the-carbon-footprint-of-machine-learning-training/
- Buckley S. (2024) IEA Study Sees AI, Cryptocurrency Doubling Data Center Energy Consumption by 2026, https://www.datacenterfrontier.com/energy/article/33038469/iea-study-sees-ai-cryptocurrency-doubling-data-center-energy-consumption-by-2026