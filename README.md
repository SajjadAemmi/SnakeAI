# Snake AI
Snake game written using the PyGame 

Trained with PyTorch

![model arch](Screenshot.png)


# Train
Use the following command for generate eight neighborhoods csv dataset:

```
python generate_dataset_8_direction.py --count 1000000
```

Use the following command for train model:

```
python train.py --epochs 8
```


# Run
Use the following command for run game with machine learning inference:

```
python main_machine_learning.py --weights ./weights/snake.pt
```

Use the following command for run game with classic rule based AI inference:

```
python main_rule_based.py
```
