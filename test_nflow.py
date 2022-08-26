from models.normalizing_flow import NormalizingFlow
from models.distribution import Distribution, NormalDistribution, TargetDistribution1, TargetDistribution2
import tensorflow as tf
import os
import re
from typing import List
import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np

AXIS_INVISIBLE = True
FIGURE_DIR = './figure'
NORMALIZING_FLOW_LAYER_NUM = 16
ITERATION = 30000
BATCH_SIZE = 1024
NUMBER_OF_SAMPLES_FOR_VISUALIZE = 2048
SAVE_FIGURE_INTERVAL = 1000
FIGURE_SIZE = 6


def camel_to_snake(camel_str: str) -> str:
    """Convert camel case string to snake case string"""
    snake_str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    snake_str = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_str).lower()
    return snake_str

def save_loss_values(loss_values: List, distribution_name: str, loss_interval=1) -> None:
    result_figure_dir = os.path.join(FIGURE_DIR, f'result_{camel_to_snake(distribution_name)}')
    if not os.path.exists(result_figure_dir):
        os.makedirs(result_figure_dir)
    
    plt.figure(figsize=(6, 4))
    plt.plot([loss_interval * i for i in range(len(loss_values))], loss_values)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(result_figure_dir, f'loss.png'))
    plt.clf()
    plt.close()
    

def save_result(z_k_value: ndarray, iteration: int, distribution_name: str) -> None:
    """Save samples from Normalizing Flow"""
    result_figure_dir = os.path.join(FIGURE_DIR, f'result_{camel_to_snake(distribution_name)}')
    if not os.path.exists(result_figure_dir):
        os.makedirs(result_figure_dir)
    
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    plt.scatter(z_k_value[:, 0], z_k_value[:, 1], alpha=0.6)
    if AXIS_INVISIBLE:
        plt.tick_params(bottom=False,left=False,right=False,top=False)
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)    
    plt.savefig(os.path.join(result_figure_dir, f'{iteration}iteration.png'))
    plt.clf()
    plt.close()

target_distribution = TargetDistribution1

normalizing_flow = NormalizingFlow(K=NORMALIZING_FLOW_LAYER_NUM)

z_0, log_q_0 = normalizing_flow.get_placeholder()
z_k, log_q_k = normalizing_flow.forward(z_0, log_q_0)
loss = normalizing_flow.calc_loss(z_k, log_q_k, target_distribution)
trainer = normalizing_flow.get_trainer(loss)

print('Calculation graph constructed')

loss_values = []
    
with tf.Session() as sess:
    print('Session Start')
    sess.run(tf.global_variables_initializer())
    print('All variables initialized')
    print(f'Training Start (number of iterations: {ITERATION})')

    for iteration in range(ITERATION + 1):
        z_0_batch = NormalDistribution.sample(BATCH_SIZE)
        log_q_0_batch = np.log(NormalDistribution.calc_prob(z_0_batch))
        _, loss_value = sess.run([trainer, loss], {z_0:z_0_batch, log_q_0:log_q_0_batch})
        loss_values.append(loss_value)

        if iteration % 100 == 0:
            iteration_digits = len(str(ITERATION))
            print(f'Iteration:  {iteration:<{iteration_digits}}  Loss:  {loss_value}')

        if iteration % SAVE_FIGURE_INTERVAL == 0:
            z_0_batch_for_visualize = NormalDistribution.sample(NUMBER_OF_SAMPLES_FOR_VISUALIZE)
            log_q_0_batch_for_visualize = np.log(NormalDistribution.calc_prob(z_0_batch_for_visualize))
            z_k_value = sess.run(z_k, {z_0:z_0_batch_for_visualize, log_q_0:log_q_0_batch_for_visualize})
            save_result(z_k_value, iteration, target_distribution.__name__)
            save_loss_values(loss_values, target_distribution.__name__)
            
    print('Training Finished')
    
print('Session Closed')