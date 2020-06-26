# DOCUMENT EXTRACTION.

## Purpose:
For converting Unstructured OCR documents into Strutured key value pairs.

## Required packages:
> - Wand 
> - Pytesseract
> - Tesseract
> - Ghost script
> - Imagemagick
> - Open CV
> - Sklearn
> - Keras
> - Tensorflow
## Usage
Replace absolute path of pdf in main function of GETO2.0.py

## Key technologies used: 
> - **_Deep learning_**, 
> - **_Ensembled learning_** 

## Description of machine learning architectures.
> - **_DoT-Net_**: DoT-Net is a novel and innovative CNN architecture to classify and segment the text elements in the document.
>- **_RFClassifier_**: RFClassifier is ensembled deep learning architecture used to detect TOC pages with in the document.

## Flow Diagram of the frame work
![Alt text](/DoTNet/Framework.png?raw=true "Framework")


## CODE FOLLOW:
> - **_GETO2.0.py_** is the interface for our framework.
> - **Segmentation.py** is the module for DoT-Net. This function is used in GETO2.0.py
> - **TOCclassifier.py** is the module to detect the TOC in the document. This function is used in GETO2.0.py
> - **TESSARACT.py** is used for extract text entites from detected blocks of text in segmentation.py. This function is used in TOCclassifier.py
> - **BlockParsing.py** is used to extract TOC entites form TOCs pages detected in TOCclassifier. This function is used in Segementation.py

## CODE FLOW:
![Alt text](/DoTNet/Codepipeline.png?raw=true "Code flow")
## Detail description of code:
##### GET02.0.py:

GETO2.0 is the main interface of our framework. Each page in input pdf file is converted to image using wand library. This convert image is checked for TOC by using TOCclassifier (We only check for TOC in first **_N_** pages).
- [x] Pages detected as ToC.
  - ##### TOCclassifer.py : TOCclassifier check the pages for TOC. If the page is classified as TOC then we use **tesseract.py** to extract the Text information for TOC and append in a list.
    - ##### tesseract.py: Tesseract.py uses the **pytesseract** (python wrapper of tesseract. Tesseract is a text extraction framework from images), for extracting text from TOC.
- [x] Page detected as Non-ToC.
  - **Note**: Pages after the first **_N_** is also considered as Non-ToC.
  - ##### Segmentation.py : Segmentation does mutiple tasks.
    > - It segements the pages by using image morophology methods and counter functions, to find the Conneted Comments (Blocks).
    > - A sliding window is passed over these Connected Components to generate **100 * 100** size tiles (DoT-Net takes 100 * 100 tiles as input to classify.
    > - A data dulipcation or augmentation is performed on blocks which are less than 100 * 100 (especially for headings the blocks size will be less than 100 * 100), to avoid the data missing issue. 
    > - Now this is 100 * 100 are classifed using DoT-Net. 
    > - After patch classification we use majorty voting to predict the label of block.
    > - If block label is text. Then we use **_blockparsing.py_** to extract the text from blocks.
      > - **_Note_**: Our DoT-Net can detect other classes such as Table, Image, Mathematical Expressions, and Line drawings, but for this project we are only focused on Text.
    > - Blockparsing.py uses pytesseract to extract the text.
    > - Append the extracted text in list
 - [x] Text from TOC and remaining PDF document is extarcted and appended in respective lists.
    > - After Extracting text from TOC and remaining pdf document and appended in list. 
    > - we use fuzzy matching and regular expression matchings techniques to create JSON files



