# Action routing evaluation

This evaluation measures three offline components against a manually labeled JSONL dataset:

- intent classification accuracy and macro-F1;
- catalog normalization exact match, positive-field accuracy, and false-positive rate;
- first planned tool-route accuracy derived from the resolved `AgentAction`.

The current dataset contains 200 cases:

- 24 chat;
- 64 product search;
- 20 product detail;
- 24 availability;
- 24 quote;
- 20 order list;
- 24 multi-device scenario continuation.

Eighty-four product-search and order cases contain field-level normalization labels.

Run from the repository root with the configured model and GearMate PostgreSQL catalog:

```bash
python evals/evaluate_action_routing.py
```

The latest report is written to `evals/results/action_routing_latest.json`. Each prediction is retained so metric changes can be audited case by case.

The tool metric is a routing metric, not a RentFlow availability metric. It checks whether the resolved action would route to the expected first tool; it does not call RentFlow or count tool HTTP execution success.
