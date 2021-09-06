import csv
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from yarr.agents.agent import ScalarSummary, HistogramSummary, ImageSummary, \
    VideoSummary
from torch.utils.tensorboard import SummaryWriter
import wandb 

class LogWriter(object):

    def __init__(self,
                 logdir: str,
                 tensorboard_logging: bool,
                 csv_logging: bool,
                 wandb_logging: bool,
                 ):
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._wandb_logging = wandb_logging
        os.makedirs(logdir, exist_ok=True)
        if tensorboard_logging:
            self._tf_writer = SummaryWriter(logdir)
        if csv_logging:
            self._prev_row_data = self._row_data = OrderedDict()
            self._csv_file = os.path.join(logdir, 'data.csv')
            self._field_names = None

    def add_scalar(self, i, name, value):
        if self._tensorboard_logging:
            self._tf_writer.add_scalar(name, value, i)
        # if self._wandb_logging:
        #     wandb.log({name: value, name+'_step': i})
        if self._csv_logging:
            if len(self._row_data) == 0:
                self._row_data['step'] = i
            self._row_data[name] = value.item() if isinstance(
                value, torch.Tensor) else value

    def add_summaries(self, i, summaries):
        wandb_log = {'Train step': i}
        for summary in summaries:
            try:
                if isinstance(summary, ScalarSummary):
                    self.add_scalar(i, summary.name, summary.value)
                    wandb_log.update( {summary.name: summary.value} )
                elif isinstance(summary, ImageSummary):
                    v = (summary.value if summary.value.ndim == 3 else
                             summary.value[0])
                    if self._tensorboard_logging:
                        self._tf_writer.add_image(summary.name, v, i)
                    wandb_log.update( {summary.name: wandb.Image(v)} )
                elif isinstance(summary, VideoSummary):
                    v = (summary.value if summary.value.ndim == 5 else
                             np.array([summary.value]))
                    if self._tensorboard_logging:
                        self._tf_writer.add_video(
                            summary.name, v, i, fps=summary.fps)
                    wandb_log.update( {summary.name: wandb.Video(v, fps=summary.fps)} )
                # elif self._tensorboard_logging:
                #     # if isinstance(summary, HistogramSummary):
                #     #     self._tf_writer.add_histogram(
                #     #         summary.name, summary.value, i)
                #     if isinstance(summary, ImageSummary):
                #         # Only grab first item in batch
                #         v = (summary.value if summary.value.ndim == 3 else
                #              summary.value[0])
                #         self._tf_writer.add_image(summary.name, v, i)
                #     elif isinstance(summary, VideoSummary):
                #         # Only grab first item in batch
                #         v = (summary.value if summary.value.ndim == 5 else
                #              np.array([summary.value]))
                #         self._tf_writer.add_video(
                #             summary.name, v, i, fps=summary.fps)
            except Exception as e:
                logging.error('Error on summary: %s' % summary.name)
                raise e
        if self._wandb_logging:
            wandb.log(wandb_log)
            
    def end_iteration(self):
        if self._csv_logging and len(self._row_data) > 0:
            with open(self._csv_file, mode='a+') as csv_f:
                names = self._field_names or self._row_data.keys()
                writer = csv.DictWriter(csv_f, fieldnames=names)
                if self._field_names is None:
                    writer.writeheader()
                else:
                    if not np.array_equal(self._field_names, self._row_data.keys()):
                        # Special case when we are logging faster than new
                        # summaries are coming in.
                        missing_keys = list(set(self._field_names) - set(
                            self._row_data.keys()))
                        for mk in missing_keys:
                            self._row_data[mk] = self._prev_row_data[mk]
                self._field_names = names
                writer.writerow(self._row_data)
            self._prev_row_data = self._row_data
            self._row_data = OrderedDict()

    def close(self):
        if self._tensorboard_logging:
            self._tf_writer.close()
        if self._wandb_logging:
            wandb.finish() 
