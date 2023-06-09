#GOAL#

To Develop a deep learning model using Keras that accurately classifies images as either 'dog' or 'cat' based on the input image data.
The model should be trained on a large dataset of labeled images of dogs and cats and should achieve a high accuracy level in differentiating between the two classes.

Uscases are:

Pet adoption platforms: The model can be used to automatically classify images of pets uploaded by users on adoption platforms, making it easier for potential adopters to search for specific types of animals.

Veterinary clinics: The model can assist veterinary clinics by automatically classifying images of animals and helping with the initial diagnosis. For example, it can determine whether an image shows a dog or a cat, enabling faster triage or guiding preliminary assessments.

Animal shelters: The model can be utilized in animal shelters to automate the process of classifying incoming animals. This can aid in population monitoring, tracking adoption rates for specific species, and identifying trends in animal intake.

Social media platforms: Social media platforms that involve sharing pet photos can leverage this model to automatically categorize uploaded images, enhance user experiences, and provide relevant recommendations to users based on their interests.

Security systems: The model can be integrated into security systems to differentiate between dogs and cats in surveillance footage. This can help filter out non-threatening activities from alerts or provide insights into the presence of specific animals in an area.

E-commerce platforms: Online stores specializing in pet products can use the model to automatically categorize pet images shared by customers. This can improve search functionality, personalize product recommendations, and enhance the overall shopping experience.


#Image Source#

https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip


#Image standarzation#

Images get standardized to (192,192,3)

from PIL import Image
import os

# Set the target size
target_size = (192, 192, 3)

# Set the path to the folder containing the images
folder_path = 'Dataset/PetImages/train/cat'

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):  # Update with the file extensions you have
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        
        # Resize the image
        resized_image = image.resize(target_size[:2])
        
        # Convert the image to RGB if it's not already
        if resized_image.mode != 'RGB':
            resized_image = resized_image.convert('RGB')
        
        # Save the resized image, overwriting the original
        resized_image.save(image_path)