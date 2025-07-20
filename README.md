<div align="center">
  
## Text-to-Remote-Sensing-Image retrieval beyond RGB sources

[**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Lorenzo Vaiani**](https://scholar.google.com/citations?user=JQVjbNEAAAAJ&hl=it&oi=sra)<sup>1</sup> · [**Giuseppe Gallipoli**]()<sup>1</sup> · [**Luca Cagliero**]()<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

<a href="https://arxiv.org/abs/2507.10403"><img src='https://img.shields.io/badge/Paper-red?style=flat&logo=arxiv&labelColor=black' alt='Dataset'></a>
<a href="https://huggingface.co/datasets/DarthReca/crisislandmark"><img src='https://img.shields.io/badge/CrisisLandMark-yellow?style=flat&logo=huggingface&labelColor=black' alt='Dataset'></a>
<a href="https://huggingface.co/collections/DarthReca/closp-687ce68f6ff7502cfb07ba3c"><img src='https://img.shields.io/badge/CLOSP-yellow?style=flat&logo=huggingface&labelColor=black' alt='Models'></a>
</div>

This paper introduces **CLOSP** (Contrastive Language Optical SAR Pretraining), a novel model that unifies text, multispectral optical, and SAR data into a shared latent space, moving beyond traditional RGB-only retrieval. We also present 
**CrisisLandMark**, a new benchmark corpus with over 647,000 Sentinel-1 (SAR) and Sentinel-2 (optical) images paired with structured annotations for land cover and crisis events. Our work demonstrates that jointly training on optical and SAR data significantly improves semantic retrieval performance, especially for complex SAR imagery.

*REPOSITORY IN CONSTRUCTION*

## Getting Started
Install the dependencies of the *requirements.txt* file. Make sure to edit the config files in the `configs/` folder. Then simply run *train.py*

## Corpus

The CrisisLandMark corpus is a new, large-scale dataset designed for Text-to-Remote-Sensing-Image Retrieval (T2RSIR) that moves beyond the limitations of RGB-only data.
- Size: Contains over 647,000 paired examples of satellite imagery and structured textual annotations.
- Focus: Enables the development and evaluation of retrieval models that can jointly interpret text, multispectral optical (Sentinel-2), and SAR (Sentinel-1) data.
- Public Availability: The corpus is publicly available on [Hugging Face](https://huggingface.co/datasets/DarthReca/crisislandmark).

### Data Modalities

- Sentinel-1 (SAR): Synthetic Aperture Radar (SAR) imagery, which provides all-weather, day-and-night structural information. Comprises 338,342 images in the corpus. Each image has two radiometric channels: VV (Vertical transmit-Vertical receive) and VH (Vertical transmit-Horizontal receive).
- Sentinel-2 (Optical): Multispectral imagery (MSI) that captures rich spectral signatures beyond visible light. Comprises 308,461 images in the corpus. Each image product contains 12 spectral bands, including ultra-blue, visible, near-infrared, and short-wave infrared.

### Annotations 

To provide a consistent framework, annotations from all sources are harmonized into a unified label space. 
This system is based on the 9 classes from Dynamic World, supplemented with specific keywords for crisis events:

Trees, Crops, Shrub and Scrub, Water, Grass, Built, Flooded Vegetation, Bare, Snow and Ice, Flooded Area, Earthquake Damage, Burned Area	

## Models

| Model | Vision Backbone | Text Backbone | Weights |
| :--- | :--- | :--- | :--- |
| **CLOSP-RN** | ResNet-50 | MiniLM  | [Link](https://huggingface.co/DarthReca/CLOSP-RN) |
| **CLOSP-VS** | ViT-S | MiniLM  | [Link](https://huggingface.co/DarthReca/CLOSP-VS) |
| **CLOSP-VL** | ViT-L | MiniLM  | [Link](https://huggingface.co/DarthReca/CLOSP-VL) |
| **GEOCLOSP** | ResNet-50 | MiniLM | [Link](https://huggingface.co/DarthReca/GeoCLOSP-RN) |

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@misc{cambrin2025texttoremotesensingimageretrievalrgbsources,
      title={Text-to-Remote-Sensing-Image Retrieval beyond RGB Sources}, 
      author={Daniele Rege Cambrin and Lorenzo Vaiani and Giuseppe Gallipoli and Luca Cagliero and Paolo Garza},
      year={2025},
      eprint={2507.10403},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.10403}, 
}
```
