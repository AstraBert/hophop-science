---
layout: post
title: "BrAIn: next generation neurons?"
tags: [AI, Brain, Open-Science]
comments: false
---

## A game-changing publication?

![_config.yml]({{ site.baseurl }}/images/pexels-nadezhda-moryak-9162028.jpg)

11th December 2023 may seem a normal day to most people, and if you are one of them, prepare to be surprised: a breakthrough publication was indeed issued on _Nature Electronic_, and it happened to put the foundations for a field that will be crucial in the next future - the title? "Brain organoid reservoir computing for artificial intelligence".

I bet that everyone can somewhat grasp the idea behind the title, but it is worth introducing some key concepts for those who may be unfamiliar with the biological notions behind the brain.

## Biological concepts

- **Neuron**: Neurons are the building blocks of a brain. They are small cells with a round central body called _soma_, where all the biological activity is carried out, some input branched-like structures (the _dendrites_), that receive signals from neighbor neurons, and an output wire-like _axon_, along which the bioelectric signal is conducted. 
- **Synapsis**: As the Greek etymology underlines, synapses are "points of contact": the end of the axon is indeed enlarged into a "synaptic button", from which neurotransmitters are released following a bioelectrical signal. On the other end, there is a dendrite which, despite not touching the synaptic button, is really close to it, separated only but what is called the "synaptic slit": it receives the neurotransmitter, evoking a bioelectrical response
- **Action potential**: Neurons transmit their bioelectric signals through an all-or-nothing event, known as _action potential_. The action potential originates in the point where the axon emerges from the soma, known as _axon hillock_, because of a series of ionic exchanges across the neural membrane. When the axon hillock reaches a voltage threshold, the neuron fires, and the shape of the electric signal curve is always the same: what determines the differences among sensations, emotions and memories is the _frequency_ of the firing. The action potential is transmitted along the axon, and the regions which were surpassed by the signal become refractory to other stimulation for a short time, ensuring a one-way transmission of the action potential (from axon hillock to dendrites).
- **Synaptic plasticity**: this phenomenon encompasses modifications of the quantity of released neurotransmitters, number of receptors... which are performed on a synapsis to reinforce or weaken it, based on how much and how well it is used. 
- **Neuroplasticity**: Neuroplasticity is the phenomenon through which brain neurons rearrange to optimize the way they respond to external stimuli.
- **Organoid**: An organoid is an arrangement of living cells into complex structures which mimic the functioning of an organ. They are used for simulations and experiments.

## Explaining the breakthrough

Now that we have all the concepts we need, let's dive into understanding what happened in the paper we mentioned in the first paragraph, and in general what is going on in the field. 

### 1. The core: organoid intelligence

_Organoid Intelligence_ (OI) is a dynamic and growing field in bio-computing, whose base idea is to harness the power of human neurons, arranged into brain organoids, to speed up computing, ease training and provide a cheap and reliable alternative to artificial neural networks, to run AI algorithms and perform tasks. The ultimate aim of this field is to build a "wet-ware" (as opposed to the already-existent hardware), a concept that the mentioned paper describes as _brainoware_, with which we'll be able to implement brain-machine interfaces and dramatically increase our power.

### 2. The findings: speech recognition and comparison with ANNs

In "Brain organoid reservoir computing for artificial intelligence", the team behind the paper built a small brain organoid, loaded onto a multielectrode array (MAE) chip. 

The organoid was trained to recognized the speech of 8 people with 240 recordings, showing different neural patterns of activation when different people were speaking and achieving a 78% accuracy in recognizing them. This may sound pretty unsurprising, unless you consider the size of the training dataset: 240 recordings are a really small-sized set of data, considered that AI algorithms would need thousands of examples to achieve similar accuracy scores.

After that, some other tests were performed, but one was really important, because it encompassed the comparison among the ONN (organoid neural netowrk), ANNs with a Long-Short Term Memory (LSTM) unit and ANNs without it. Brain cells were trained through impulses for four days (four 'epochs') on solving a 200-data point map. ONN outperformed ANNs without LSTMs, while ANNs+LSTM only were able to prove a little bit more accurate than the organoid only with 50 epochs of training, which means that ONNs yield similar results to their artificial counterparts with >90% less training.

### 3. Advantages and limitations

There are big advantages linked to OI:

- Train for less time and with less data, obtain high accuracy
- We can use it to explain how brain works and to take a look into brain degenerative diseases like Alzheimer's disease
- High volumes of managed and output data

Despite the promising perspectives, there are still some obstacles we need to overcome:

- Current organoids are scarcely persistent, we need something more durable and reliable 
- We need to adapt machine-brain interfaces to smoother and more biological-friendly structure, in order to seamlessly connect and tune brain input/outputs with external machines.
- We have to scale up our algorithms and models to handle the huge volume of data that organoids will be able to manage

## Conclusion

Organoid Intelligence is undoubtedly the forefront of biocomputing, which will be able to revolutionize the way we understand and (probably) even think of our brain, unlocking novel and unexpected discoveries on how we learn notions and shape our memory. On the other hand, it will provide a powerful hardware, which will capture huge conceptual, computational and representational power in small brain-like engines, reducing learning times and expenses for our new AI models. All of this, obviously, is subjected to the condition that we invest resources and time in building new organoids, algorithms and data facilities: the future of _brAIn_ is close, we just need to put some effort to reach it. 

## References

- Cai, H., Ao, Z., Tian, C. et al. Brain organoid reservoir computing for artificial intelligence. Nat Electron 6, 1032–1039 (2023). https://doi.org/10.1038/s41928-023-01069-w
- Tozer L, ‘Biocomputer’ combines lab-grown brain tissue with electronic hardware, https://www.nature.com/articles/d41586-023-03975-7
- Smirnova L, Caffo BS, Gracias DH, Huang Q, Morales Pantoja IE, Tang B, Zack DJ, Berlinicke CA, Boyd JL, Harris TD, Johnson EC, Kagan BJ, Kahn J, Muotri AR, Paulhamus BL, Schwamborn JC, Plotkin J, Szalay AS, Vogelstein JT, Worley PF and Hartung T. Organoid intelligence (OI): the new frontier in biocomputing and intelligence-in-a-dish. Front Sci (2023) 1:1017235. doi: 10.3389/fsci.2023.1017235
