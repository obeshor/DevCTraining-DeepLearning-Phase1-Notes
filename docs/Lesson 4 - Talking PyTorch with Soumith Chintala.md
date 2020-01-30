# 4 - Talking PyTorch with Soumith Chintala

#### Origins of PyTorch
*  Soumith Chintala always wanted to be a visual effects artist at least when he started his undergrad and then he interned at a place and they said he's not good enough
* He was good at programming since he was a kid
* He try to find the next most magical thing and that was computer vision
* He had to find a professor in India which is really hard to afford who's doing this kind of stuff and it was just like one or two and he spent six months with the professor's lab
* He started picking up some things then went to CMU tried his hand at robotics and then finally landed at NYU and Yann LeCun's lab doing deep learning
* He got to NYU, he've been working on building up tooling for Deep Learning.
* He worked on this project called EB learn which was like two generations before in terms of deep learning framework timelines.
* Then came around torch which is written by a few people
* He started getting pretty active and helping people out using torch and then developing a torch
* He wanted to do his research and then helped other people do their research well
* At some point we decided that we needed a new tool because  as the field moves, the tools that you need to makes research progress change as well.
* He went about building PyTorch mostly because They had a really stressful project that was really large and hard to build
* PyTorch had an interesting development experience, they started with just three of them and then they got other people interested.
* About eight or nine people joined in part-time just adding feature and then slowly and steadily they started giving access to other people.
* Every week they would give access to about like ten people  for like about 6 monts and then in Jan be released by doors to the public

#### Debugging and Designing PyTorch
* If you have a non-contiguous tensor and sent it through a linear layer it will just give you garbage
* A trade-off there where the readability comes at a cost of it being a little bit slow
* The autograd and the entire internal autograd engine is all written in python.
* It should be very imperative very usable very pythonic but at the same time as fast as any other framework
* The consequences of that was like large parts of PyTorch live in C++ except whatever is user-facing is still in python.
* You can attach your debugger you can print, those are still very very hackable.

#### From Research to Production
* They gave it a bunch of researchers and they took a rapid feedback from them and improve the product before it became mature so the core design of PyTorch is  very researcher friendly
* PyTorch is designed with users and just their feedback in mind
* PyTorch especially in its latest version sort of does also add features that make it easier to deploy models to production
* For production, you want  to export your whole model and do like C++ runtime or you want to run things like you'd take your network and quantize it and start running into 32-bit, 8 bits or 4 bits
* They built PyTorch event geared for production is you do research but when you want it to be production ready you just add functional annotations to your model which are like these one-liners that are top of a function. PyTorch will parse your model, your python program itself

#### Hybrid Frontend
* They called a new programming model hybrid front-end because you can make parts of a model like compiled parts of my model and gives you the best of both worlds.
* The hybrid font end is allowing you to just of switch in between python and basicall like C++ representation.
* In the short-term, the JIT compiler in PyTorch is to make sure we can export everything to production ready.
* In th long-term, they are going to make the parts of model non trivialy faster by fusing operations, making more of the memory bandwith bound operations into compute bond operations

#### Cutting-edge Applications in PyTorch
* One paper written by one person Andrew Brock it was called ``smash`` where one neural network would generate the weights that would be powered for another neural network.
* Angela Fan and her collaborators publised ``hierarchical story generation`` so you would see a story with like hey I want a story of a boy swimming in a pond and then it would actually like generate a story that's interesting with that plot
* Openly available github repositories, it's also just like very readable of work where you look at something you can clearly see like here are the inputs here is what's happening as far as it being transformed and here are the desired outputs

#### User Needs and Adding Features
* When they're exploring new ideas they don't want to be seeing like a 10x drop in performance
* Online courses they want more interactive tutorials like based on a Python notebooks 
* Some widgets they want first-class integration with colab for free GPU.
* Users have been asking for support for Google TPUs, Cloud providers (Amazon, Azure made  pyTorch a first-class citizen). Google announced support not onl for Pytorch a first-class citizen, also for TPU support  and TensorBoard integration.

#### PyTorch and the Facebook Product
* Facebook has the AI space, they have Facebook AI Research where they publish research  onto archive with  open datasets, to peer review.
* They also have a huge set of needs for products at Facebook whether it's our camera enhancements or whether it is our machine translation or whether it's our accessibility interfaces or our integrity filtering. They need tools that can do all of these.
* Their mission is to do AI research in the open and advance humanity using AI

#### The Future of PyTorch
* The next thing he was thinking : deep learning itself is becoming a very pervasive and essential competent in many other fields (health care, data, computational chemistry, Neurobiology, Neuroscience). So they would build a cute package which they coud relate to.

#### Learning More in AI
* Students are yet trying to get into the field of deep learning either to apply it to their own stuff or just to learn the concepts it's very important to make sure you do it from day one; He encourages people to do projects, review the projects and give feedback
* His only advice to people is to make sure you do lesser but like do it hands-on







