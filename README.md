# Sound AI progress
Tracking states of the arts and recent results (bibliography) on sound AI topics and audio tasks. <i> Feel free to create PRs for new results!</i>

Inspired by [wer_are_we](https://github.com/syhw/wer_are_we) and [are_we_there_yet](http://rodrigob.github.io/are_we_there_yet/build/)

## Sound AI or Audio Analytics
Sound AI or Audio Analytics focuses on analyzing and understanding audio signals captured by digital devices, with numerous applications in health & wellbeing, environmental sensing, urban living, and the creative sector. 


## Table of Contents

* [Sound Event Classification](#sound-event-classification)
* [Acoustic Scene Classification](#acoustic-scene-classification)
* [Audio Captioning](#audio-captioning)
* [Text to Audio Retrieval](#text-to-audio-retrieval)
* [Audio to Text Retrieval](#audio-to-text-retrieval)
* [Music Classification](#music-classification)

## Sound Event Classification

### AudioSet
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST [ensemble]</sub> | <sub>0.496</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection**</sub> | <sub>Transformer model with hierarchical structure and token-semantic modules [ensemble]</sub> | <sub>0.487</sub> | <sub>[chen2022](https://arxiv.org/pdf/2202.00874.pdf)</sub> | <a href="https://github.com/RetroCirce/HTS-Audio-Transformer">:scroll:</a>   |
| <sub>**BEATs: Audio Pre-Training with Acoustic Tokenizers**</sub> | <sub>iterative audio pre-training framework to learn bidirectional encoder representation from audio transformers</sub> | <sub>0.486</sub> | <sub>[chen22](https://arxiv.org/pdf/2212.09058.pdf)</sub> | <a href="https://github.com/microsoft/unilm/tree/master/beats">:scroll:</a> |
| <sub>**AST: Audio Spectrogram Transformer**</sub> | <sub>Pure Attention Model Pretrained on AudioSet [ensemble]</sub> | <sub>0.485</sub> | <sub>[gong2021](https://arxiv.org/pdf/2104.01778.pdf)</sub> | <a href="https://github.com/YuanGongND/ast">:scroll:</a> |
| <sub>**Masked Autoencoders that Listen**</sub> | <sub>extension of image-based Masked Autoencoders (MAE) to self-supervised representation learning from audio spectrograms</sub> | <sub>0.473</sub> | <sub>[huang2022](https://arxiv.org/pdf/2207.06405.pdf)</sub> | <a href="https://github.com/facebookresearch/AudioMAE">:scroll:</a> |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST [non-ensemble]</sub> | <sub>0.471</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection**</sub> | <sub>Transformer model with hierarchical structure and token-semantic modules [non-ensemble]</sub> | <sub>0.471</sub> | <sub>[chen2022](https://arxiv.org/pdf/2202.00874.pdf)</sub> | <a href="https://github.com/RetroCirce/HTS-Audio-Transformer">:scroll:</a>   |
| <sub>**AST: Audio Spectrogram Transformer**</sub> | <sub>Pure Attention Model Pretrained on AudioSet [non-ensemble]</sub> | <sub>0.459</sub> | <sub>[gong2021](https://arxiv.org/pdf/2104.01778.pdf)</sub> | <a href="https://github.com/YuanGongND/ast">:scroll:</a> |
| <sub>**PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition**</sub> | <sub>CNN models trained on AudioSet</sub> | <sub>0.439</sub> | <sub>[kong2019](https://arxiv.org/pdf/1912.10211.pdf)</sub> | <a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">:scroll:</a> |
| <sub>**Conformer-Based Self-Supervised Learning for Non-Speech Audio Tasks**</sub> | <sub>Conformer-based self-supervised learning</sub> | <sub>0.415</sub> | <sub>[srivastava2022](https://arxiv.org/pdf/2110.07313.pdf)</sub> | |

### FSD50K
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST</sub> | <sub>0.653</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**</sub> | <sub>CLAP trained on LAION 650k collection with feature fusion and caption augmentation </sub> | <sub> 0.649 </sub> | <sub>[wu2022](https://arxiv.org/pdf/2211.06687.pdf)</sub> | <a href=" https://github.com/LAION-AI/CLAP">:scroll:</a> |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>0.5859</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP</sub> | <sub>0.4308</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |

### ESC50
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**BEATs: Audio Pre-Training with Acoustic Tokenizers**</sub> | <sub>iterative audio pre-training framework to learn bidirectional encoder representation from audio transformers</sub> | <sub>98.1%</sub> | <sub>[chen22](https://arxiv.org/pdf/2212.09058.pdf)</sub> | <a href="https://github.com/microsoft/unilm/tree/master/beats">:scroll:</a> |
| <sub>**Masked Autoencoders that Listen**</sub> | <sub>Image-based MAE for audio spectrograms</sub> | <sub>97.4%</sub> | <sub>[huang2022](https://arxiv.org/pdf/2207.06405.pdf)</sub> | <a href="https://github.com/facebookresearch/AudioMAE">:scroll:</a> |
| <sub>**HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection**</sub> | <sub>Transformer model with hierarchical structure and token-semantic modules</sub> | <sub>97.00%</sub> | <sub>[chen2022](https://arxiv.org/pdf/2202.00874.pdf)</sub> | <a href="https://github.com/RetroCirce/HTS-Audio-Transformer">:scroll:</a>   |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST</sub> | <sub>96.8%</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>96.70%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**AST: Audio Spectrogram Transformer**</sub> | <sub>Pure Attention Model Pretrained on AudioSet</sub> | <sub>95.70%</sub> | <sub>[gong2021](https://arxiv.org/pdf/2104.01778.pdf)</sub> | <a href="https://github.com/YuanGongND/ast">:scroll:</a> |
| <sub>**Connecting the Dots between Audio and Text without Parallel Data through Visual Knowledge Transfer**</sub> | <sub>A Transformer model pretrained w/ visual image supervision</sub> | <sub>95.70%</sub> | <sub>[zhao2022](https://arxiv.org/pdf/2112.08995.pdf)</sub> | <a href="https://github.com/zhaoyanpeng/vipant">:scroll:</a> |
| <sub>**A Sequential Self Teaching Approach for Improving Generalization in Sound Event Recognition**</sub> | <sub>Multi-stage sequential learning with knowledge transfer from Audioset</sub> | <sub>94.10%</sub> | <sub>[kumar2020](https://arxiv.org/pdf/2007.00144.pdf)</sub> |  |
| <sub>**Efficient End-to-End Audio Embeddings Generation for Audio Classification on Target Applications**</sub> | <sub>CNN model pretrained on AudioSet</sub> | <sub>92.32%</sub> | <sub>[lopez-meyer2021](https://ieeexplore.ieee.org/document/9414229)</sub> |  |
| <sub>**Urban Sound Tagging using Multi-Channel Audio Feature with Convolutional Neural Networks**</sub> | <sub>Pretrained model with multi-channel features</sub> | <sub>89.50%</sub> | <sub>[kim2020](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_JHKim_21_t5.pdf)</sub> | <a href="https://github.com/JaehunKim-DeepLearning/Dcase2020_Task5">:scroll:</a> |
| <sub>**An Ensemble of Convolutional Neural Networks for Audio Classification**</sub> | <sub>CNN ensemble with data augmentation</sub> | <sub>88.65%</sub> | <sub>[nanni2020](https://arxiv.org/pdf/2007.07966.pdf)</sub> | <a href="https://github.com/LorisNanni/Ensemble-of-Convolutional-Neural-Networks-for-Bioimage-Classification">:scroll:</a> |
| <sub>**Environmental Sound Classification on the Edge: A Pipeline for Deep Acoustic Networks on Extremely Resource-Constrained Devices**</sub> | <sub>CNN model (ACDNet) with potential compression</sub> | <sub>87.1%</sub> | <sub>[mohaimenuzzaman2021](https://arxiv.org/pdf/2103.03483.pdf)</sub> | <a href="https://anonymous.4open.science/r/71077d05-6666-43a7-ae73-ec5ce2bef91b/">:scroll:</a> |
| <sub>**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**</sub> | <sub>CNN with filterbanks learned using convolutional RBM + fusion with GTSC and mel energies</sub> | <sub>86.50%</sub> | <sub>[sailor2017](https://pdfs.semanticscholar.org/f6fd/1be38a2d764d900b11b382a379efe88b3ed6.pdf)</sub> |  |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP</sub> | <sub>85.95%</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |
| <sub>**AclNet: efficient end-to-end audio classification CNN**</sub> | <sub>CNN with mixup and data augmentation</sub> | <sub>85.65%</sub> | <sub>[huang2018](https://arxiv.org/pdf/1811.06669.pdf)</sub> |  |
| <sub>**On Open-Set Classification with L3-Net Embeddings for Machine Listening Applications**</sub> | <sub>x-vector network with openll3 embeddings</sub> | <sub>85.00%</sub> | <sub>[wilkinghoff2020](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0000800.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)) + data augmentation + Between-Class learning</sub> | <sub>84.90%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Novel Phase Encoded Mel Filterbank Energies for Environmental Sound Classification**</sub> | <sub>CNN working with phase encoded mel filterbank energies (PEFBEs), fusion with Mel energies</sub> | <sub>84.15%</sub> | <sub>[tak2017](https://www.researchgate.net/profile/Dharmesh_Agrawal/publication/320733074_Novel_Phase_Encoded_Mel_Filterbank_Energies_for_Environmental_Sound_Classification/links/5a084c780f7e9b68229c8947/Novel-Phase-Encoded-Mel-Filterbank-Energies-for-Environmental-Sound-Classification.pdf)</sub> |  |
| <sub>**Knowledge Transfer from Weakly Labeled Audio using Convolutional Neural Network for Sound Events and Scenes**</sub> | <sub>CNN pretrained on AudioSet</sub> | <sub>83.50%</sub> | <sub>[kumar2017](https://arxiv.org/pdf/1711.01369.pdf)</sub> | <a href="https://github.com/anuragkr90/weak_feature_extractor">:scroll:</a> |
| <sub>**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**</sub> | <sub>CNN with filterbanks learned using convolutional RBM + fusion with GTSC</sub> | <sub>83.00%</sub> | <sub>[sailor2017](https://pdfs.semanticscholar.org/f6fd/1be38a2d764d900b11b382a379efe88b3ed6.pdf)</sub> |  |
| <sub>**Deep Multimodal Clustering for Unsupervised Audiovisual Learning**</sub> | <sub>CNN + unsupervised audio-visual learning</sub> | <sub>82.60%</sub> | <sub>[hu2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Hu_Deep_Multimodal_Clustering_for_Unsupervised_Audiovisual_Learning_CVPR_2019_paper.pdf)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>Fusion of GTSC & TEO-GTSC with CNN</sub> | <sub>81.95%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)) + Between-Class learning</sub> | <sub>81.80%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| :headphones: <sub>***Human accuracy***</sub> | <sub>Crowdsourcing experiment in classifying ESC-50 by human listeners</sub> | <sub>81.30%</sub> | <sub>[piczak2015a](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-dataset">:scroll:</a> |
| <sub>**Objects that Sound**</sub> | <sub>*Look, Listen and Learn* (L3) network ([arandjelovic2017a](https://arxiv.org/pdf/1705.08168.pdf)) with stride 2, larger batches and learning rate schedule</sub> | <sub>79.80%</sub> | <sub>[arandjelovic2017b](https://arxiv.org/pdf/1712.06651.pdf)</sub> |  |
| <sub>**Look, Listen and Learn**</sub> | <sub>8-layer convolutional subnetwork pretrained on an audio-visual correspondence task</sub> | <sub>79.30%</sub> | <sub>[arandjelovic2017a](https://arxiv.org/pdf/1705.08168.pdf)</sub> |  |
| <sub>**Learning Environmental Sounds with Multi-scale Convolutional Neural Network**</sub> | <sub>Multi-scale convolutions with feature fusion (waveform + spectrogram)</sub> | <sub>79.10%</sub> | <sub>[zhu2018](https://arxiv.org/pdf/1803.10219.pdf)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>GTSC with CNN</sub> | <sub>79.10%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)) + data augmentation</sub> | <sub>78.80%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Unsupervised Filterbank Learning Using Convolutional Restricted Boltzmann Machine for Environmental Sound Classification**</sub> | <sub>CNN with filterbanks learned using convolutional RBM</sub> | <sub>78.45%</sub> | <sub>[sailor2017](https://pdfs.semanticscholar.org/f6fd/1be38a2d764d900b11b382a379efe88b3ed6.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>Baseline CNN ([piczak2015b](http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf)) + Batch Normalization + Between-Class learning</sub> | <sub>76.90%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>TEO-GTSC with CNN</sub> | <sub>74.85%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>EnvNet-v2 ([tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf))</sub> | <sub>74.40%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Soundnet: Learning sound representations from unlabeled video**</sub> | <sub>8-layer CNN (raw audio) with transfer learning from unlabeled videos</sub> | <sub>74.20%</sub> | <sub>[aytar2016](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)</sub> | <a href="https://github.com/cvondrick/soundnet">:scroll:</a> |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>18-layer CNN on raw waveforms ([dai2016](https://arxiv.org/pdf/1610.00087.pdf)) + Between-Class learning</sub> | <sub>73.30%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Novel Phase Encoded Mel Filterbank Energies for Environmental Sound Classification**</sub> | <sub>CNN working with phase encoded mel filterbank energies (PEFBEs)</sub> | <sub>73.25%</sub> | <sub>[tak2017](https://www.researchgate.net/profile/Dharmesh_Agrawal/publication/320733074_Novel_Phase_Encoded_Mel_Filterbank_Energies_for_Environmental_Sound_Classification/links/5a084c780f7e9b68229c8947/Novel-Phase-Encoded-Mel-Filterbank-Energies-for-Environmental-Sound-Classification.pdf)</sub> |  |
| <sub>**Classifying environmental sounds using image recognition networks**</sub> | <sub>16 kHz sampling rate, GoogLeNet on spectrograms (40 ms frame length)</sub> | <sub>73.20%</sub> | <sub>[boddapati2017](https://www.sciencedirect.com/science/article/pii/S1877050917316599)</sub> | <a href="https://github.com/bkasvenkatesh/Classifying-Environmental-Sounds-with-Image-Networks">:scroll:</a> |
| <sub>**Learning from Between-class Examples for Deep Sound Recognition**</sub> | <sub>Baseline CNN ([piczak2015b](http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf)) + Batch Normalization</sub> | <sub>72.40%</sub> | <sub>[tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>Fusion of MFCC & TEO-GTCC with GMM</sub> | <sub>72.25%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Learning environmental sounds with end-to-end convolutional neural network (EnvNet)**</sub> | <sub>Combination of spectrogram and raw waveform CNN</sub> | <sub>71.00%</sub> | <sub>[tokozume2017a](http://www.mi.t.u-tokyo.ac.jp/assets/publication/LEARNING_ENVIRONMENTAL_SOUNDS_WITH_END-TO-END_CONVOLUTIONAL_NEURAL_NETWORK-poster.pdf)</sub> |  |
| <sub>**Novel TEO-based Gammatone Features for Environmental Sound Classification**</sub> | <sub>TEO-GTCC with GMM</sub> | <sub>68.85%</sub> | <sub>[agrawal2017](http://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347591.pdf)</sub> |  |
| <sub>**Classifying environmental sounds using image recognition networks**</sub> | <sub>16 kHz sampling rate, AlexNet on spectrograms (30 ms frame length)</sub> | <sub>68.70%</sub> | <sub>[boddapati2017](https://www.sciencedirect.com/science/article/pii/S1877050917316599)</sub> | <a href="https://github.com/bkasvenkatesh/Classifying-Environmental-Sounds-with-Image-Networks">:scroll:</a> |
| <sub>**Very Deep Convolutional Neural Networks for Raw Waveforms**</sub> | <sub>18-layer CNN on raw waveforms</sub> | <sub>68.50%</sub> | <sub>[dai2016](https://arxiv.org/pdf/1610.00087.pdf), [tokozume2017b](https://openreview.net/forum?id=B1Gi6LeRZ)</sub> | <a href="https://github.com/philipperemy/very-deep-convnets-raw-waveforms">:scroll:</a> |
| <sub>**Classifying environmental sounds using image recognition networks**</sub> | <sub>32 kHz sampling rate, GoogLeNet on spectrograms (30 ms frame length)</sub> | <sub>67.80%</sub> | <sub>[boddapati2017](https://www.sciencedirect.com/science/article/pii/S1877050917316599)</sub> | <a href="https://github.com/bkasvenkatesh/Classifying-Environmental-Sounds-with-Image-Networks">:scroll:</a> |
| <sub>**WSNet: Learning Compact and Efficient Networks with Weight Sampling**</sub> | <sub>SoundNet 8-layer CNN architecture with 100x model compression</sub> | <sub>66.25%</sub> | <sub>[jin2017](https://openreview.net/forum?id=H1I3M7Z0b)</sub> |  |
| <sub>**Soundnet: Learning sound representations from unlabeled video**</sub> | <sub>5-layer CNN (raw audio) with transfer learning from unlabeled videos</sub> | <sub>66.10%</sub> | <sub>[aytar2016](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)</sub> | <a href="https://github.com/cvondrick/soundnet">:scroll:</a> |
| <sub>**WSNet: Learning Compact and Efficient Networks with Weight Sampling**</sub> | <sub>SoundNet 8-layer CNN architecture with 180x model compression</sub> | <sub>65.80%</sub> | <sub>[jin2017](https://openreview.net/forum?id=H1I3M7Z0b)</sub> |  |
| <sub>**Soundnet: Learning sound representations from unlabeled video**</sub> | <sub>5-layer CNN trained on raw audio of ESC-50 only</sub> | <sub>65.00%</sub> | <sub>[aytar2016](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)</sub> | <a href="https://github.com/cvondrick/soundnet">:scroll:</a> |
| <sub>:bar_chart: **Environmental Sound Classification with Convolutional Neural Networks** - ***CNN baseline***</sub> | <sub>CNN with 2 convolutional and 2 fully-connected layers, mel-spectrograms as input, vertical filters in the first layer</sub> | <sub>64.50%</sub> | <sub>[piczak2015b](http://karol.piczak.com/papers/Piczak2015-ESC-ConvNet.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-convnet">:scroll:</a> |
| <sub>**auDeep: Unsupervised Learning of Representations from Audio with Deep Recurrent Neural Networks**</sub> | <sub>MLP classifier on features extracted with an RNN autoencoder</sub> | <sub>64.30%</sub> | <sub>[freitag2017](https://arxiv.org/pdf/1712.04382.pdf)</sub> | <a href="https://github.com/auDeep/auDeep">:scroll:</a> |
| <sub>**Classifying environmental sounds using image recognition networks**</sub> | <sub>32 kHz sampling rate, AlexNet on spectrograms (30 ms frame length)</sub> | <sub>63.20%</sub> | <sub>[boddapati2017](https://www.sciencedirect.com/science/article/pii/S1877050917316599)</sub> | <a href="https://github.com/bkasvenkatesh/Classifying-Environmental-Sounds-with-Image-Networks">:scroll:</a> |
| <sub>**Classifying environmental sounds using image recognition networks**</sub> | <sub>CRNN</sub> | <sub>60.30%</sub> | <sub>[boddapati2017](https://www.sciencedirect.com/science/article/pii/S1877050917316599)</sub> | <a href="https://github.com/bkasvenkatesh/Classifying-Environmental-Sounds-with-Image-Networks">:scroll:</a> |
| <sub>**Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks**</sub> | <sub>3-layer CNN with vertical filters on wideband mel-STFT (*median accuracy*)</sub> | <sub>*56.37%*</sub> | <sub>[huzaifah2017](https://arxiv.org/pdf/1706.07156.pdf)</sub> |  |
| <sub>**Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks**</sub> | <sub>3-layer CNN with square filters on wideband mel-STFT (*median accuracy*)</sub> | <sub>*54.00%*</sub> | <sub>[huzaifah2017](https://arxiv.org/pdf/1706.07156.pdf)</sub> |  |
| <sub>**Soundnet: Learning sound representations from unlabeled video**</sub> | <sub>8-layer CNN trained on raw audio of ESC-50 only</sub> | <sub>51.10%</sub> | <sub>[aytar2016](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)</sub> | <a href="https://github.com/cvondrick/soundnet">:scroll:</a> |
| <sub>**Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks**</sub> | <sub>5-layer CNN with square filters on wideband mel-STFT (*median accuracy*)</sub> | <sub>*50.87%*</sub> | <sub>[huzaifah2017](https://arxiv.org/pdf/1706.07156.pdf)</sub> |  |
| <sub>**Comparison of Time-Frequency Representations for Environmental Sound Classification using Convolutional Neural Networks**</sub> | <sub>5-layer CNN with vertical filters on wideband mel-STFT (*median accuracy*)</sub> | <sub>*46.25%*</sub> | <sub>[huzaifah2017](https://arxiv.org/pdf/1706.07156.pdf)</sub> |  |
| :bar_chart: <sub>***Baseline - random forest***</sub> | <sub>Baseline ML approach (MFCC & ZCR + random forest)</sub> | <sub>44.30%</sub> | <sub>[piczak2015a](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-dataset">:scroll:</a> |
| <sub>**Soundnet: Learning sound representations from unlabeled video**</sub> | <sub>Convolutional autoencoder trained on unlabeled videos</sub> | <sub>39.90%</sub> | <sub>[aytar2016](http://papers.nips.cc/paper/6146-soundnet-learning-sound-representations-from-unlabeled-video.pdf)</sub> | <a href="https://github.com/cvondrick/soundnet">:scroll:</a> |
| :bar_chart: <sub>***Baseline - SVM***</sub> | <sub>Baseline ML approach (MFCC & ZCR + SVM)</sub> | <sub>39.60%</sub> | <sub>[piczak2015a](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-dataset">:scroll:</a> |
| :bar_chart: <sub>***Baseline - k-NN***</sub> | <sub>Baseline ML approach (MFCC & ZCR + k-NN)</sub> | <sub>32.20%</sub> | <sub>[piczak2015a](http://karol.piczak.com/papers/Piczak2015-ESC-Dataset.pdf)</sub> | <a href="https://github.com/karoldvl/paper-2015-esc-dataset">:scroll:</a> |

### US8K
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**AudioCLIP: Extending CLIP to Image, Text and Audio**</sub> | <sub>incorporates the ESResNeXt audio-model into the CLIP framework using the AudioSet dataset </sub> | <sub>90.07%</sub> | <sub>[guzhov2021](https://arxiv.org/pdf/2106.13043.pdf)</sub> | <a href="https://github.com/AndreyGuzhov/AudioCLIP">:scroll:</a> |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>87.96%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP</sub> | <sub>81.01%</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |

### VocalSound
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>97.95%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**Vocalsound: A Dataset for Improving Human Vocal Sounds Recognition**</sub> | <sub>EfficientNetB0</sub> | <sub> 90.5% </sub> | <sub>[gong2022](https://arxiv.org/pdf/2205.03433.pdf)</sub> | <a href="https://github.com/YuanGongND/vocalsound">:scroll:</a> |

## VGGSound
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**Slow-Fast Auditory Streams For Audio Recognition**</sub> | <sub>two-stream convolutional network for audio recognition</sub> | <sub> 54.4% </sub> | <sub>[kazakos2022](https://arxiv.org/pdf/2103.03516.pdf)</sub> | <a href="https://github.com/ekazakos/auditory-slow-fast">:scroll:</a> |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP</sub> | <sub>46.63%</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |


## Acoustic Scene Classification

## Audio Captioning

### AudioCaps
| <sub>Title</sub> | <sub>Notes</sub> | <sub>SPIDEr</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**Audio Captioning Transformer**</sub> | <sub>Transformer network based on an encoder-decoder architecture</sub> | <sub> 0.426 </sub> | <sub>[mei2021](https://arxiv.org/pdf/2107.09817.pdf)</sub> | <a href="https://github.com/XinhaoMei/ACT">:scroll:</a> |

### Clotho
| <sub>Title</sub> | <sub>Notes</sub> | <sub>SPIDEr</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**WaveTransformer: A Novel Architecture for Audio Captioning Based on Learning Temporal and Time-Frequency Information**</sub> | <sub>two-branch audio encoder for learning temporal and local time-frequency information</sub> | <sub> 0.182 </sub> | <sub>[tran2020](https://arxiv.org/pdf/2010.11098.pdf)</sub> | <a href="https://github.com/an-tran528/wavetransformer">:scroll:</a> |

## Text to Audio Retrieval

### AudioCaps
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP@10</sub> | <sub>R@1</sub>| <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**</sub> | <sub>CLAP trained on LAION 650k collection with feature fusion and caption augmentation </sub> | | <sub> 36.7 </sub> | <sub>[wu2022](https://arxiv.org/pdf/2211.06687.pdf)</sub> | <a href=" https://github.com/LAION-AI/CLAP">:scroll:</a> |
| <sub>**Audio Retrieval with Natural Language Queries: A Benchmark Study**</sub> | <sub>MoE, CE and MMT used</sub> |  | <sub> 36.1 </sub> |<sub>[koepke2022](https://arxiv.org/pdf/2203.15537.pdf)</sub> | <a href=" https://github.com/akoepke/audio-retrieval-benchmark">:scroll:</a> |
| <sub>**Audio Retrieval with WavText5K and CLAP Training**</sub> | <sub>CLAP training with WavText5K added</sub> | <sub> 49.45 </sub> | <sub>  34.69 </sub> |<sub>[deshmukh2022](https://arxiv.org/pdf/2209.14275.pdf)</sub> | <a href=" https://github.com/microsoft/WavText5K">:scroll:</a> |
| <sub>**On metric learning for audio-text cross-modal retrieval**</sub> | <sub>Metric learning objectives for audio retrieval</sub> |  | <sub> 33.9 </sub> |<sub>[mei2022](https://arxiv.org/pdf/2203.15537.pdf)</sub> | <a href=" https://github.com/XinhaoMei/audio-text_retrieval">:scroll:</a> |

### Clotho
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP@10</sub> | <sub>R@1</sub>| <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**</sub> | <sub>CLAP trained on LAION 650k collection with feature fusion and caption augmentation </sub> | | <sub> 18.2 </sub> | <sub>[wu2022](https://arxiv.org/pdf/2211.06687.pdf)</sub> | <a href=" https://github.com/LAION-AI/CLAP">:scroll:</a> |
| <sub>**Audio Retrieval with WavText5K and CLAP Training**</sub> | <sub>CLAP training with WavText5K added</sub> | <sub> 27.12 </sub> | <sub>  16.75 </sub> | <sub>[deshmukh2022](https://arxiv.org/pdf/2209.14275.pdf)</sub> | <a href=" https://github.com/microsoft/WavText5K">:scroll:</a> |
| <sub>**On metric learning for audio-text cross-modal retrieval**</sub> | <sub>Metric learning objectives for audio retrieval</sub> |  | <sub> 14.4 </sub> |<sub>[mei2022](https://arxiv.org/pdf/2203.15537.pdf)</sub> | <a href=" https://github.com/XinhaoMei/audio-text_retrieval">:scroll:</a> |
| <sub>**Audio Retrieval with Natural Language Queries: A Benchmark Study**</sub> | <sub>MoE, CE and MMT used</sub> |  | <sub> 6.7 </sub> |<sub>[koepke2022](https://arxiv.org/pdf/2112.09418.pdf)</sub> | <a href=" https://github.com/akoepke/audio-retrieval-benchmark">:scroll:</a> |


## Audio to Text Retrieval

### AudioCaps
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP@10</sub> | <sub>R@1</sub>| <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**</sub> | <sub>CLAP trained on LAION 650k collection with feature fusion and caption augmentation </sub> | | <sub> 46.8 </sub> | <sub>[wu2022](https://arxiv.org/pdf/2211.06687.pdf)</sub> | <a href=" https://github.com/LAION-AI/CLAP">:scroll:</a> |
| <sub>**Audio Retrieval with WavText5K and CLAP Training**</sub> | <sub>CLAP training with WavText5K added</sub> | <sub> 30.81 </sub> | <sub>  41.91 </sub> |<sub>[deshmukh2022](https://arxiv.org/pdf/2209.14275.pdf)</sub> | <a href=" https://github.com/microsoft/WavText5K">:scroll:</a> |
| <sub>**On metric learning for audio-text cross-modal retrieval**</sub> | <sub>Metric learning objectives for audio retrieval</sub> |  | <sub> 39.6 </sub> |<sub>[mei2022](https://arxiv.org/pdf/2203.15537.pdf)</sub> | <a href=" https://github.com/XinhaoMei/audio-text_retrieval">:scroll:</a> |
| <sub>**Audio Retrieval with Natural Language Queries: A Benchmark Study**</sub> | <sub>MoE, CE and MMT used</sub> |  | <sub> 39.6 </sub> |<sub>[koepke2022](https://arxiv.org/pdf/2112.09418.pdf)</sub> | <a href=" https://github.com/akoepke/audio-retrieval-benchmark">:scroll:</a> |

### Clotho
| <sub>Title</sub> | <sub>Notes</sub> | <sub>mAP@10</sub> | <sub>R@1</sub>| <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- | :--- |
| <sub>**Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**</sub> | <sub>CLAP trained on LAION 650k collection with feature fusion and caption augmentation </sub> | | <sub> 25.7 </sub> | <sub>[wu2022](https://arxiv.org/pdf/2211.06687.pdf)</sub> | <a href=" https://github.com/LAION-AI/CLAP">:scroll:</a> |
| <sub>**Audio Retrieval with WavText5K and CLAP Training**</sub> | <sub>CLAP training with WavText5K added</sub> | <sub> 13.65 </sub> | <sub> 20.00 </sub> |<sub>[deshmukh2022](https://arxiv.org/pdf/2209.14275.pdf)</sub> | <a href=" https://github.com/microsoft/WavText5K">:scroll:</a> |
| <sub>**On metric learning for audio-text cross-modal retrieval**</sub> | <sub>Metric learning objectives for audio retrieval</sub> |  | <sub> 16.9 </sub> |<sub>[mei2022](https://arxiv.org/pdf/2203.15537.pdf)</sub> | <a href=" https://github.com/XinhaoMei/audio-text_retrieval">:scroll:</a> |
| <sub>**Audio Retrieval with Natural Language Queries: A Benchmark Study**</sub> | <sub>MoE, CE and MMT used</sub> |  | <sub> 7.2 </sub> |<sub>[koepke2022](https://arxiv.org/pdf/2112.09418.pdf)</sub> | <a href=" https://github.com/akoepke/audio-retrieval-benchmark">:scroll:</a> |

## Music Classification

### GTZAN Genres
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>91.3%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>88.3%</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition**</sub> | <sub>CNN models trained on AudioSet [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>86.0%</sub> | <sub>[kong2019](https://arxiv.org/pdf/1912.10211.pdf)</sub> | <a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">:scroll:</a> |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>74.8%</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |

### GTZAN Music Speech
| <sub>Title</sub> | <sub>Notes</sub> | <sub>Accuracy</sub> | <sub>Paper</sub> | <sub>Code</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>**CLAP: Learning Audio Concepts From Natural Language Supervision**</sub> | <sub>CNN model pretrained by natural language supervision</sub> | <sub>100%</sub> | <sub>[elizalde2022](https://arxiv.org/pdf/2206.04769.pdf)</sub> | <a href="https://github.com/microsoft/CLAP">:scroll:</a> |
| <sub>**PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition**</sub> | <sub>CNN models trained on AudioSet [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>99.23%</sub> | <sub>[kong2019](https://arxiv.org/pdf/1912.10211.pdf)</sub> | <a href="https://github.com/qiuqiangkong/audioset_tagging_cnn">:scroll:</a> |
| <sub>**PaSST: Efficient Training of Audio Transformers with Patchout**</sub> | <sub>drops out some of the input patches during training of AST [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>97.69%</sub> | <sub>[koutini22](https://arxiv.org/pdf/2110.05069.pdf)</sub> | <a href="https://github.com/kkoutini/PaSST">:scroll:</a> |
| <sub>**Wav2CLIP: Learning Robust Audio Representations From CLIP**</sub> | <sub>Distilling from CLIP [[HEAR Challenge]](https://hearbenchmark.com/hear-leaderboard.html)</sub> | <sub>94.55%</sub> | <sub>[wu2021](https://arxiv.org/pdf/2110.11499.pdf)</sub> | <a href="https://github.com/descriptinc/lyrebird-wav2clip">:scroll:</a> |


## Glossary
SED: Sound Event Detection <br>
ASC: Acoustic Scene Classification
