In this method, we take inspiration from [panotti repository](https://github.com/drscotthawley/panotti) and train our data.

Once we train our model, to test it out, we use test.py script with the name of audio file as arg.
Once our model detects whether the file has a whale call or not, we can then find where the position of whale call occured in the sample.

For this, we use two methods and evaluate their trade-offs:

### Template matching and extraction:
The inspiration of Template matching comes from [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html). We can very well use this method for extracting features and to find the location of whale up-call.

Template Matching is a method for searching and finding the location of a template image in a larger image. We use this template image as a [whale-call](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_1/assets/whale_template.png).

We take these below images to find the location of the whale call:


<p align="center">
  <img  src=assets/whale1.png/>
  Whale Call
</p>


<p align="center">
  <img  src=assets/whale2.png/>
  Whale Call
</p>


<p align="center">
  <img  src=assets/nonwhale1.png/>
  Non-Whale Call
</p>


After the use of [temp_match.py](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_1/temp_match.py) script, we see the following results:


<p align="center">
  <img  src=assets/Figure1.png/>
  Whale Call (Start coordinate= 374 End coordinate= 599)
</p>

<p align="center">
  <img  src=assets/Figure_1.png/>
  Whale Call (Start coordinate= 387 End coordinate= 612)
</p>

<p align="center">
  <img  src=assets/Figure_2.png/>
 Non-Whale Call (Start coordinate= 753 End coordinate= 978)
</p>

**These start and end-coordinates can be then used to match with the spectrogram to find the exact start and end-time of orca calls**

After the Template Extraction method, we see the following results:
[This](https://github.com/ZER-0-NE/OrcaCNN-Demo/blob/master/Method_1/temp_extract.py) script extracts and plots all the templates, showing 3 images per row:

1. The original image (from where the template is extracted)
2. The image after being pre-processed
3. The template extracted, where the background is blue for whale sounds and 
red for non-whale sounds

<p align="center">
  <img  src=assets/whale_ext1.png/>
</p>


<p align="center">
  <img  src=assets/nonwhale_ext1.png/>
</p>


<p align="center">
  <img  src=assets/whale_ext2.png/>
</p>


<p align="center">
  <img  src=assets/nonwhale_ext2.png/>
</p>


### Frequency Metrics using bins:


![](assets/Fig2.png)

![](assets/Fig3.png)



