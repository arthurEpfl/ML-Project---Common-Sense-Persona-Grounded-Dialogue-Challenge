![CPDC Banner](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1131/341d077b29bc8be92ec6.png)

# [Commonsense Persona-grounded Dialogue Challenge](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/problems/task-1-commonsense-dialogue-response-generation) - Task 1 - Starter kit
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the CPD Challenge (Task 1) **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

# Table of Contents

- [Commonsense Persona-grounded Dialogue Challenge - Task 1 - Starter kit](#commonsense-persona-grounded-dialogue-challenge---task-1---starter-kit)
- [Table of Contents](#table-of-contents)
- [Competition Overview](#competition-overview)
  - [Task 1: Commonsense Dialogue Response Generation](#task-1-commonsense-dialogue-response-generation)
- [Getting Started](#getting-started)
- [How to write your own model?](#how-to-write-your-own-model)
- [How to start participating?](#how-to-start-participating)
  - [Setup](#setup)
  - [How do I specify my software runtime / dependencies?](#how-do-i-specify-my-software-runtime--dependencies)
  - [What should my code structure be like?](#what-should-my-code-structure-be-like)
  - [How to make a submission?](#how-to-make-a-submission)
- [Other Concepts](#other-concepts)
    - [Evaluation Metrics](#evaluation-metrics)
    - [Time and compute constraints](#time-and-compute-constraints)
  - [Local Evaluation](#local-evaluation)
  - [Note about Dummy test data](#note-about-dummy-test-data)
  - [Contributing](#contributing)
- [📎 Important links](#-important-links)


#  Competition Overview

This challenge is an opportunity for researchers and machine learning enthusiasts to test their skills on the challenging tasks of Commonsense Dialogue Response Generation (Task1) and Commonsense Persona Knowledge Linking (Task2) for persona-grounded dialogue.

Research on dialogue systems has been around for a long time, but thanks to Transformers and Large Language Models (LLM), conversational AI has come a long way in the last five years, becoming more human-like. On the other hand, it is still challenging to collect natural dialogue data for research and to benchmark which models ultimately perform the best because there is no definitive assessment data or metrics, and the comparisons are often within a limited amount of models.

We contribute to the research and development of current state-of-the-art dialogue systems, by crafting high quality human-human dialogues for model testing, and providing a common benchmarking venue by hosting this CPDC 2023 competition.

The competition aims to see the best approach among state-of-the-art participant models on an evaluation dataset of natural conversation. The submitted systems will be evaluated on a new Commonsense Persona-grounded Dialogue dataset. To this end, we first created several persona profiles, similar to [ConvAI2](https://arxiv.org/abs/1902.00098v1), with a natural personality based on a commonsense persona-grounded knowledge graph ([PeaCoK](https://aclanthology.org/2023.acl-long.362/)†) newly released on [ACL 2023](https://2023.aclweb.org/), and allowing us to obtain naturally related persona sentences. Furthermore, based on that persona, we created a natural dialogue between two people and prepared a sufficient amount of dialogue data for evaluation.

The **Commonsense Persona-grounded Dialogue (CPD)** Challenge hosts one track on **Commonsense Dialogue Response Generation (Task 1)** and one track on **Commonsense Persona Knowledge Linking (Task 2)**. Independent leaderboards are set for the two tracks, each featuring a separate prize pool. In either case, participants may use any learning data. In Task 1, participants will submit dialogue response generation systems. We will evaluate them on the prepared persona-grounded dialogue dataset mentioned above. In Task 2, participants will submit systems linking knowledge to a dialogue. This task is designed in the similar spirit of [ComFact](https://aclanthology.org/2022.findings-emnlp.120/), which is released along with the published paper in [EMNLP 2022](https://2022.emnlp.org/). We will evaluate them by checking if the linking of persona-grounded knowledge can be judged successfully on the persona-grounded dialogue dataset.

† [PeaCoK: Persona Commonsense Knowledge for Consistent and Engaging Narratives](https://aclanthology.org/2023.acl-long.362/)  **(ACL2023 Outstanding Paper Award)**

## [Task 1: Commonsense Dialogue Response Generation](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/problems/task-1-commonsense-dialogue-response-generation)

Participants will submit dialogue response generation systems. We do not provide a training dataset, and participants may use any datasets which they want to use. We provide [a baseline model](https://github.com/Silin159/PersonaChat-BART-PeaCoK), which can be tested on the [ConvAI2 PERSONA-CHAT](https://arxiv.org/abs/1902.00098v1) dataset, so that you can see what the problem of this task is. We will evaluate submitted systems on the persona-grounded dialogue dataset. The dialogues in the evaluation dataset have persona sentences similar to the PersonaChat dataset, but the number of persona sentences for a person is more than five sentences. The major part of the persona is derived from the [PeaCoK](https://github.com/Silin159/PeaCoK) knowledge graph.

![](https://lh7-us.googleusercontent.com/rJQ8H9qwj2LtKPswZCiPRD01_cD440o1zymhDPAHsNt4fQaAv9IsYerhlHJmL-2rH88t7WbDRkr2uEKg7MosnLEtWEBMdRYOr9SQD3B09Xjn5vmOMg_6hXXZBMW9uXQaHaF69aWefon64brDQuIBfjU)


#  Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/problems/task-1-commonsense-dialogue-response-generation).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/commonsense-persona-grounded-dialogue-challenge-2023/commonsense-persona-grounded-dialogue-challenge-task-1-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# How to write your own model?

We recommend that you place the code for all your models in the `agents/` directory (though it is not mandatory). You should implement the following

- `generate_responses` - This function is called to generate the response of a conversation given two persona information.

**Add your agent name in** `agent/user_config.py`, this is what will be used for the evaluations.
  
An example are provided in `agent/dummy_agent.py`

# How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/commonsense-persona-grounded-dialogue-challenge-2023/commonsense-persona-grounded-dialogue-challenge-task-1-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```h
    git clone git@gitlab.aicrowd.com:aicrowd/challenges/commonsense-persona-grounded-dialogue-challenge-2023/commonsense-persona-grounded-dialogue-challenge-task-1-starter-kit
    ```

3. **Install** competition specific dependencies!
    ```
    cd commonsense-persona-grounded-dialogue-challenge-task-1-starter-kit
    pip install -r requirements.txt
    ```

4. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

5. Test your model locally using `python local_evaluation.py`

6. Make a submission as described in [How to make a submission](#how-to-make-a-submission) section.

## How do I specify my software runtime / dependencies?

We accept submissions with custom runtime, so you don't need to worry about which libraries or framework to pick from.

The configuration files typically include `requirements.txt` (pypi packages), `apt.txt` (apt packages) or even your own `Dockerfile`.

An example Dockerfile is provided in [utilities/_Dockerfile](utilities/_Dockerfile) which you can use as a starting point.

You can check detailed information about setting up runtime dependencies in the 👉 [docs/runtime.md](docs/runtime.md) file.

## What should my code structure be like?

Please follow the example structure as it is in the starter kit for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── apt.txt                # Linux packages to be installed inside docker image
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Use this to check your model evaluation flow locally
├── dummy_data_task1.json  # A set of dummy conversations you can use for integration testing
└── agents                 # Place your models related code here
    ├── dummy_agent.py             # Dummy agent for example interface
    └── user_config.py              # IMPORTANT: Add your agent name here
```

Finally, **you must specify an AIcrowd submission JSON in `aicrowd.json` to be scored!**

The `aicrowd.json` of each submission should contain the following content:

```json
{
  "challenge_id": "task-1-commonsense-dialogue-response-generation",
  "authors": ["your-aicrowd-username"],
  "gpu": true,
  "description": "(optional) description about your awesome model"
}
```

This JSON is used to map your submission to the challenge - so please remember to use the correct `challenge_id` as specified above. You can modify the `authors` and `description` keys. Please DO NOT add any additional keys to `aicrowd.json` unless otherwise communicated during the course of the challenge.

# Other Concepts
### Evaluation Metrics


### Time and compute constraints

You will be provided conversations with 7 turns each in `batches of upto 50 conversations`. For each batch of conversations, the first set of turns will be provided to your model. After the response is receieved the further turns of the same conversation will be provided. Each conversation will have exactly 7 turns. Your model needs to `complete all 7 responses of 50 conversations within **1 hour**`. The number of batches of conversation your model will process will vary based on the challenge round.

Before running on the challenge dataset, your model will be run on the dummy data, as a sanity check. This will show up as the `convai-validation` phase on your submission pages. The dummy data will contain `5 conversations of 7 turns each`, your model needs to `complete the validation phase within **15 minutes**`.

Your model will be run on an AWS g5.2xlarge node. This node has **8 vCPUs, 32 GB RAM, and one Nvidia A10G GPU with 24 GB VRAM**.

Before your model starts processing conversations, it is provided an additional time upto *5 minutes* to load models or preprocess any data if needed.

## Local Evaluation

Participants can run the evaluation protocol for their model locally with or without any constraint posed by the challenge to benchmark their models privately. See `local_evaluation.py` for details. You can change it as you like, your changes to `local_evaluation.py` will **NOT** be used for the competition.

## Note about Dummy test data

The file `dummy_data_task1.json` is a dummy test dataset to test your code before submission. All dialogues in the dataset based on a same pair of persona A and persona B, but the actual test dataset for evaluation is not like this and was created based on different pairs of personas.

## Contributing

🙏 You can share your solutions or any other baselines by contributing directly to this repository by opening merge request.

- Add your implemntation as `agents/<your_agent>.py`.
- Import it in `user_config.py`
- Test it out using `python local_evaluation.py`.
- Add any documentation for your approach at top of your file.
- Create merge request! 🎉🎉🎉

# How to make a submission?

👉 Follow the instuctions provided here [docs/submission.md](docs/submission.md)

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/problems/task-1-commonsense-dialogue-response-generation

- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/discussion

- 🏆 Leaderboard: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2023/problems/task-1-commonsense-dialogue-response-generation/leaderboards
