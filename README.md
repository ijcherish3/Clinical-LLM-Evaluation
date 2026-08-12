# Evaluation of Local LLMs for Medical-Record Question Answering

## Overview

This project evaluates local large language models for
medical-record question answering using an adapted PDSQI-9
evaluation framework.

## Models Evaluated

- Qwen 2.5 1.5B
- Qwen 2.5 3B
- Llama 3.2 3B
- Gemini 3.1 Pro

## Evaluation

Responses were evaluated across:
- Accuracy
- Thoroughness
- Usefulness
- Organization
- Comprehensibility
- Succinctness

Overall quality was calculated as the mean of these six dimensions.

## Repository Contents

- `code/` — Analysis and statistical testing
- `results/` — Figures and results
- `references/` — Supporting references

## Future Work

The selected local model will be fine-tuned using the
training dataset and subsequently evaluated using the
held-out testing dataset, with the eventual goal of
on-device clinical question answering on iPhone.
