# 10 - Deploying PyTorch Models
### Lectures

#### Welcome!
* PyTorch popular in research settings due to:
  * flexibility
  * expressiveness
  * ease of development
* Adoption has been slow in industry because wasn't as useful in production environments which typically require models to run in C++

#### Installing PyTorch 1.0
* Go to [Getting Started](https://pytorch.org/get-started/locally/) page, configure and run the install command
* Minimum requirement is PyTorch 1.0 to use TorchScript and tracing features.

#### PyTorch for Production
* PyTorch 1.0 has been specifically built for making transition between developing model in Python and converting it into a module that can be loaded into a C++ environment
* __tracing__ 
  * map out the structure of model by passing an example tensor through it, 
  * behind the scene PyTorch keeping track of all the operations that being performed on the inputs. 
  * this way, it can actually build a static graph that can then be exported and loaded into C++.
  * to do this we use JIT (Just In-Time) compiler

#### Torch Script & Tracing
* See [Torch Script](https://pytorch.org/docs/stable/jit.html)
* __torch script__
  * an intermediate representation that can be compiled and serialized by the Torch Script Compiler
  * the workflows is as follows: develop model, setting hyper parameter, train, test, convert PyTorch model into Torch Script and compile to C++ representation
  * two ways of converting PyTorch model to Torch Script
    * tracing 
    * annotations

<p align="center">
  <img src="./images/lesson-9/tracing.PNG" width="50%">
</p> 

#### Annotations
* Used in control flow that don't actually work with tracing method, for example use some if statements in forward method that depend on input.
* Use `torch.jit.ScriptModule` subclass and add  `@torch.jit.script_method` decorator to convert to script module
* We can use `save` method to serialize script module to a file which can then be loaded into C++

<p align="center">
  <img src="./images/lesson-9/annotations.PNG" width="50%">
</p> 

#### PyTorch C++ API
* See [PyTorch C++ API](https://pytorch.org/cppdocs/)
* General workflow:
  * building and defining model in Python with PyTorch
  * training it there, and then once it's all trained
  * convert it to a Script Module either with tracing or annotations
  * then serialize it with the save method
  * from there we can use the C++ API to load it into a C++ application
