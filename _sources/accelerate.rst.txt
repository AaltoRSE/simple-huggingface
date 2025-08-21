Multi-GPU training with Accelerate
==================================

What is Accelerate?
###################

`Accelerate <https://huggingface.co/docs/accelerate/index>`__ is a library
designed to simplify multi-GPU training of PyTorch models.

It supports many different parallelization strategies like
Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP) and
DeepSpeed.

The main selling point of the library is that it can handle things like
model placement, dataloader division and gradient accumulation
across multiple GPUs on multiple machines with minimal configuration.

There are huge number of advanced features included in Accelerate so
here we'll focus on two aspects: how Accelerate works when training a
model with Hugging Face Trainer and how Accelerate works when you
have an existing PyTorch training loop that you want to convert to use
multi-GPU training.

Configuring Accelerate
######################

Accelerate is configured using a ``yaml``-file that sets everything
from model distribution strategy to networking settings.

Typical configuration file might looks something like this
(:download:`accelerate_config.yaml <accelerate-trainer/accelerate_config.yaml>`):

.. literalinclude:: accelerate-trainer/accelerate_config.yaml

Easiest way of creating this configuration file is to run
`accelerate config <https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config>`__
that launches series of prompts about your desired parallelization strategy.

The main command used to launch Accelerate codes, ``accelerate launch``,
has `a huge number <https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch>`__
of different arguments that can be set. All of the arguments can be
set in the configuration file, but they can also be given via command line.


Using Trainer with Accelerate
#############################

Trainer has been designed to utilize Accelerate automatically.

Let's consider the following training code
(:download:`trainer_mnist_cnn.py <accelerate-trainer/trainer_mnist_cnn.py>`) that trains a simple
CNN model based on the MNIST dataset:

.. literalinclude:: accelerate-trainer/trainer_mnist_cnn.py

This code can be launched on a single GPU with

.. code-block:: bash

   python trainer_mnist_cnn.py

Converting code that uses ``Trainer`` to multi-GPU code is quite trivial,
as ``Trainer`` has been designed to work together with Accelerate.

When launched with a configuration file like the one given above, Accelerate
would try to do a DDP setup (``distributed_type: MULTI_GPU``) with 8
GPUs (``num_processes: 8``):

.. code-block:: bash

   accelerate launch --config_file accelerate_config.yaml trainer_mnist_cnn.py

In this case the model and data are both so small, that there would not be any benefits
on using multiple GPUs.

:download:`This more complex model <accelerate-trainer/trainer_imdb_finetune.py>` that
fine-tunes a language model for sentiment classification can already see some benefits
from multi-GPU training.


Using Accelerate with PyTorch code
##################################

Let's consider the following training code that again trains a simple CNN model
on MNIST dataset, but this time has a training loop closer to regular PyTorch
(:download:`trainer_mnist_cnn.py <accelerate-pytorch/pytorch_mnist_cnn.py>`)

.. literalinclude:: accelerate-pytorch/pytorch_mnist_cnn.py

This can be converted to use Accelerate with minor changes to the original code
(:download:`trainer_mnist_cnn.py <accelerate-pytorch/accelerate_mnist_cnn.py>`):

.. literalinclude:: accelerate-pytorch/accelerate_mnist_cnn.py
   :emphasize-lines: 12,74-80,92,101,110,122

Main modifications are are setting up the
`Accelerator <huggingface.co/docs/accelerate/package_reference/accelerator>`__,
letting the Accelerate handle the model placement

.. literalinclude:: accelerate-pytorch/accelerate_mnist_cnn.py
   :lines: 74-80

and making certain that loss is propagated across all distributed
models:

.. literalinclude:: accelerate-pytorch/accelerate_mnist_cnn.py
   :lines: 94-103
   :emphasize-lines: 8

Now this training loop can be launched in the same way as the ``Trainer`` one
and it will utilize multiple GPUs:

.. code-block:: bash

   accelerate launch --config_file accelerate_config.yaml accelerate_mnist_cnn.py
