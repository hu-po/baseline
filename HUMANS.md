potentially useful commands

```bash
git clone git@github.com:hu-po/baseline.git
npx https://github.com/google-gemini/gemini-cli
npm install -g @openai/codex
sudo arp-scan --localnet
```

potentially useful prompts

```
investigate the wandb-summary.json files in the wandb folder, create a new EXPERIMENTS_GEMINI.md that contains your analysis of the current performance, compute usage, and theoretical understandig of the different hyperparameters in @sweep_config.json. End with your suggestions on how to modify the hyperparameter search space. Add two new possible values for the existing categorical hyperparameters and remove one value you think is causing issues. Make your changes to @sweep_config.json and then wait for my input before launching the next sweep.
```