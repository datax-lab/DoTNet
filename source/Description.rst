Description
=============

Machine learning architectures
----------------------------------
- DoT-Net: Novel and innovative CNN architecture to classify and segment the text elements.
- RFClassifier: Ensenmbled deep learning architecture which used to detect TOC pages.

	
Code follow for DoT-Net
--------------------------
- GETO2.0.py: The interface for our framework. Each page in input pdf file is converted to image using wand library. 
			
- Segmentation.py: The module for DoT-Net. This function is used in GETO2.0.py

- TOCclassifier.py: The module to detect the TOC in the document. This function is used in GETO2.0.py

- TESSARACT.py: Extract text entites from detected blocks of text in segmentation.py. This function is used in TOCclassifier.py

- BlockParsing.py: Extract TOC entites form TOCs pages detected in TOCclassifier. This function is used in Segementation.py
