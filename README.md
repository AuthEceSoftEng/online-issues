# Online issue assignment
This repository contains a methodology for mining Jira issues using Online Machine Learning.

## Research Overview
Issue tracking systems have vast amounts of data that can be utilized to confront
multiple challenges, such as automated issue assignment, issue priority classification,
etc. Our methodology for automated issue assignment views issues as a data stream and
and is capable of adapting to the evolving nature of software projects (e.g. changes
of people in the team, different priorities in development).

The complete description of our methodology is available at the publication:
```
"Towards Effective Issue Assignment using Online Machine Learning"
Paper sent at the 29th International Conference on Evaluation and
Assessment in Software Engineering (EASE 2025), 2025.
```

This repository contains all code and instructions required to reproduce the
findings of the above publication. If this seems helpful or relevant to your
work, you may cite it.

## Instructions
Then, you can clone this repository and set the parameters in file
`properties.py` (folders for saving data, storing results and saving graphs).
The code is actually a collection of scripts that are used to make data manipulations
one after the other (we use the numbering s0, s1, s2, ... to dictate the order of
running the scripts).
