import wandb

def wandb_init(project, entity, config=None):
    wandb.init(project=project, entity=entity)
    # wandb.config = config