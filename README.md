# TensorFlow
## in nutshell


## klasyfikacja_kwiatów.ipynb
What we have there:
- Authenticating with Kaggle using kaggle.json (3 ways!)
- kaggle dataset
- 3 ways to done with data 
  - arrays/lists
    - train_test_split
    - sklearn.metrics.classification_report
    - sklearn.metrics.confusion_matrix
    - augmentation
    - add more images to folders
  - split folders (dataset -> train, validation, test folder)
  - tf.data.Dataset / tf.keras.preprocessing.image_dataset_from_directory
     
    ```
    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    ``` 
  - keras.preprocessing.image.DirectoryIterator /ImageDataGenerator.flow_from_directory
    ```
    

    train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
    validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our test data


    train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                           class_mode='sparse')
                                                           
    val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='sparse')
                                                              

    test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=test_dir,
                                                              shuffle=False,
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='sparse')

    ```
    
- tf.keras.utils.image_dataset_from_directory without split folders
  ```
  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=123,
    validation_split=0.2,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size)

  test_ds = tf.keras.utils.image_dataset_from_directory(  # Can be validation or test but must be names "validation"...
    data_dir,
    seed=123,
    validation_split=0.2,
    subset="validation",
    image_size=(img_height, img_width),
    batch_size=batch_size)

  ```
  
  ## CNN.ipynb
  
  - Grayscale images in 10 categories - Fashion MNIST 
  - Color images in 2 categories - cat and dog
  - Color images in 5 categories - flowers
