## Purpose

Hello! This Workload implements a sophisticated medical imaging analysis system for Mount Sinai Hospital that leverages deep learning to detect pneumonia in chest X-rays. The system provides an intuitive web interface for doctors to upload X-ray images and receive real-time predictions with detailed confidence scores. Beyond simple binary classification, the system provides a comprehensive infrastructure deployed on AWS with robust security measures, real-time performance monitoring, and efficient caching mechanisms.

## Technical Architecture Overview
### Infrastructure Components

**Frontend NGINX Server (Public Subnet)**

1) t3.micro instance running Ubuntu
2) NGINX reverse proxy configuration
3) Handles SSL termination
4) Routes traffic to UI server on port 5001
5) Security group: ports 22, 80, 443, 5001, 3000, 9090, 9100

**UI Server (Private Subnet)** 

1) t3.micro instance
2) Flask web application
3) Gunicorn WSGI server
4) Handles file uploads and result display
5) Security group: ports 22, 5001, 6379, 8000, 9100

**Application Server (Private Subnet)**

1) t3.medium instance
2) Redis cache implementation
3) API endpoint processing
4) Model serving infrastructure
5) Security group: ports 22, 5000, 5001, 6379, 8000, 9100

**ML Training Server (Private Subnet)**

1) p2.xlarge instance with NVIDIA CUDA support
2) TensorFlow training environment
3) Model development and training pipeline
4) Node exporter for metrics
5) Security group: ports 22, 5000, 5001, 6379, 8000, 9100

**Monitoring Server (Public Subnet)**

1) t3.micro instance
2) Prometheus metrics collection
3) Grafana dashboards
4) Security group: ports 22, 80, 443, 5001, 3000, 9090, 9100

### Network Architecture

1) VPC with public and private subnets
2) NAT Gateway for private subnet internet access
3) Security groups with principle of least privilege
4) Private subnet CIDR: 10.0.2.0/24
5) Public subnet CIDR: 10.0.1.0/24

## How it Works:

**1. X-Ray Image Upload (Doctor Interaction)**

Step 1: The doctor accesses the pneumonia detection application through a web interface hosted on the frontend server in the public subnet.

Step 2: The doctor uploads an X-ray image via this interface, which is implemented using Flask and HTMX. This request is sent to the Nginx reverse proxy.

**2. Routing Through Nginx (Frontend Server)**

Step 3: The uploaded image request reaches the Nginx server on the frontend, which acts as a reverse proxy. Nginx forwards the request to the backend Application server in the private subnet on port 5001.

**3. Receiving the Image (Backend Server)**

Step 4: The backend server receives the image from Nginx and temporarily stores it in a dedicated directory. This server, which runs Flask and Gunicorn, handles the incoming request and prepares the data for processing.

Step 5: The backend server saves metadata and the image path in Redis. Redis acts as a NoSQL database to temporarily store the uploaded image information and track the request, enabling the system to manage multiple simultaneous requests efficiently.

**4. Processing the Image with the ML Model (ML Training Server)**

Step 6: The backend server forwards the saved image path and necessary metadata to the ML Training server in the private subnet, which hosts a highly customized pneumonia detection ResNet convolutional neural network (CNN) model.

Step 7: The ML server uses NVIDIA CUDA for GPU-accelerated processing, speeding up the analysis of the X-ray image.

Step 8: The ResNet model processes the image and makes a binary prediction: pneumonia detected (positive) or not detected (negative). It also provides a confidence score for the prediction, indicating the model’s certainty.

**6. Storing and Managing Prediction Results**

Step 9: The ML Training server sends the prediction results, including the label (positive/negative) and confidence score, back to the backend server.

Step 10: The backend server saves this prediction data in Redis to ensure fast retrieval. Redis enables the backend server to fetch and display the results promptly when the doctor checks the results page.

**7. Displaying Results on the Frontend**

Step 11: Once the prediction is stored in Redis, the backend server notifies the frontend server via Gunicorn on port 5001 that the prediction is ready.

Step 12: The doctor can view the prediction results by navigating to the webpage of the Public IP of the frontend server.

Step 13: The frontend server retrieves the prediction result from the backend server, again via Gunicorn, which fetches it from Redis and displays it on the webpage. This page includes details such as the X-ray file name, prediction label (e.g., "Pneumonia Detected"), and the confidence score.

**8. Monitoring the System (Monitoring Server)**

Step 14: Prometheus on the monitoring server continuously scrapes metrics from the backend, ML Training server, and other components. These metrics include system performance, memory usage, CPU load, and the health of key processes like Gunicorn, Redis, and Nginx.

Step 15: Prometheus provides these metrics to Grafana, where they are visualized through custom dashboards. Grafana alerts can notify engineers of any potential issues in real time, such as high load on the ML Training server or backend server failures.

**9. Data Storage and Retrieval**

Step 16: The original X-ray images are stored in Amazon S3, enabling the application to archive and retrieve previous cases. This storage integration is optional but can aid in building a dataset for further training and improving the model’s accuracy.

## System Diagram

![image](https://github.com/user-attachments/assets/2ec2b4fe-e8b8-46c5-9136-8e477ed61017)

## Troubleshooting

### Initial Model Issues
The baseline model exhibited severe bias, with a 100% pneumonia detection rate regardless of input. Key issues included:

- Overconfident predictions (>0.95 confidence for all cases)
- No differentiation between normal and pneumonia cases
- Poor feature extraction from X-ray characteristics

### Model Architecture Improvements:

**Advanced Classification Layers:**

```python
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.3)(x)
x = Dense(1, activation='sigmoid')(x)
Training Pipeline Enhancements
```

**Learning Rate Optimization:**

```python
initial_learning_rate = 1e-5
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.95,
    staircase=True
)
```

**Data Augmentation Pipeline:**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=32,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.2,
    fill_mode='nearest',
    dtype=np.float32
)
```

**Class Weight Balancing:**

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
```

## Performance Metrics

**Before Optimization:**

Accuracy: 51.2%
Precision: 48.7%
Recall: 100%
F1 Score: 65.5%

**After Optimization:**

Accuracy: 90.3%
Precision: 89.7%
Recall: 91.2%
F1 Score: 90.4%

### Model not running after transitioning from Google Colab environment to the Ubuntu P3.2xlarge:

**Problem:** Our model was trained in a google Colab Notebook, where we were able to run the model, train it on our training dataset of X-ray scans and see the resulting predictions on the rest of the scans without issue. When we tried to run the model on the Ubuntu 22.04 EC2 P3.2xlarge, however, We got an error indicated that the model yielded an unexxpected result: 

```python
TypeError: generator yielded an element that did not match the expected structure. The expected structure was (tf.float32, tf.float32, tf.float32), but the yielded element was (array([[[[0.04313726, 0.04313726, 0.04313726],
         [0.04313726, 0.04313726, 0.04313726],
         [0.04313726, 0.04313726, 0.04313726],
```

**Solution:** TensoFlow was expecting tuples containing three elements as the resulting structure from our model predictions, but our train_generator originally only yielded results as tuples with 2 elements: an array for the predictions, and an for the labels of those predictions. In order to rectify this error, we had to:

1): Add `dtype=np.float32,` to our train and val generators to ensure the datatypes were numpy 32 floats as expected of TensorFlow, and 
2) Create a function that defined a new generator that added a third element to our resulting tuples, that we then wrapped around our train and val generators:

```python
def generator_wrapper(generator):
    while True:
        x, y = next(generator)  # Use next() directly on the generator
        yield (x, y, np.ones(y.shape))

train_generator_wrapped = generator_wrapper(train_generator)
val_generator_wrapped = generator_wrapper(val_generator)
```
the `generator_wrapper` adds a third, placeholder element of one's in the shape of a y that has no bearing on our models ability to detect pnuemonia nad make predictions - it just serves to meet the Tensorflow requirement of three elements in a tuple, so that we may train the model and accurately detect for Pneumonia. The wrapper also had the added benefit of solving an issue that we did encounter when training the model in the Google Colab Notebook - the model would be unable to complete every other Epoch during training due to "running out of data". This was because the data yield from the generator was in an unexpected format, and correcting the format resolved the issue, and helped us achieve a 90% F1 Score. 

### Optimizations

1) **Model Fine-Tuning:** Regularly retrain the model with new data and refine class weights to adapt to any shifts in pneumonia vs. normal case frequency, ensuring accuracy over time.

2) **Load Balancing:** Implement an Application Load Balancer to distribute requests evenly if traffic volume increases, maintaining response times and model availability. This would allow us to extend the ML predictions for Pneumonia scans to other hospitals within the Mount Sinai network, with the ALB efficiently distrubting incoming requests so every doctor can maintain availability to the model.

3) **Auto-scaling:** Configure auto-scaling for the ML and Application servers, allowing resources to scale based on user demand, optimizing cost and performance. Were we to implement the expansion plan spoken of above, the ML and Application servers would need to be in autoscaling groups to ensure they can always meet demand.

4) **Multi-AZ Deployment:** Continuing our push towards expanding to partner hospitals within the Mount Sinai Network, we would need to deploy the application across multiple Availability Zones, ensuring redundancy and avaiability should something happen to a part of the network in one AZ. Load Balancing and Auto-Scaling would work in concert to ensure the application remains avaiable to all doctors in need of access.

5) **AWS Elasticache for Redis:** To complete our multi-AZ expansion for partner hospitals, we would need to employ AWS Elasticache for Redis to ensure that there is a Redis node in each of our AZ's, accessible by any doctor who needs to access the scans and the results of the Pneumonia Detection Model. Elasticache for Redis would allow us to enable Mutli-AZ with Auto Failover, so should something happen to a Redis node in one AZ, a replica node in another AZ can take it's place, ensuring minimal downtime and eliminating any single points of failure.

6) **CloudWatch Implementation**: Switching from Prometheus and Grafana to the AWS managed service CloudWatch can lead to quicker, automated metric gatering and analysis on our ML and Application servers, and can lead to less downtime should the servers encountering an issue or otherwise be rendered offline. Finetuned alarms would ensure that we are always aware of the health of our servers.

## Conclusion

In this workload, we set out to develop a reliable and high-performing deep learning model that aids doctors at Mount Sinai Hospital in diagnosing pneumonia from X-ray images. Our solution salvaged an initially unreliable pneumonia detection model by incorporating advanced training techniques—such as class weighting, learning rate scheduling, and the addition of complex neural layers—to achieve an impressive 90% accuracy. Through thoughtful infrastructure provisioning and robust monitoring, the deployment is both secure and efficient, supporting real-time diagnostic assistance. Future optimizations will continue to enhance the model’s performance, scalability, and resilience in a clinical setting, making this a sustainable solution for medical image analysis.

### Documentation

**Pneumonia ML Results Webpage**


![image](https://github.com/user-attachments/assets/8a96c713-990f-4772-ba6a-3bbab2f9534a)


**Positive Pneumonia Scan of Uploaded X-ray**


![pneumonia_example](https://github.com/user-attachments/assets/737970ac-053b-4025-845a-ff152561253e)


**Negative Pneumonia Scan of Uploaded X-ray**


![normal_xray_2](https://github.com/user-attachments/assets/9db394d8-a6e8-4efb-bab6-7ddc0190e473)


**ML Pneumonia Detection Model Training**


![Real_Best Pnu_AI](https://github.com/user-attachments/assets/e283698f-1496-422f-af51-4b27c0edad60)


**Pneumonia Detection Model Confusion Matrix Scores**


![confusion_matrix](https://github.com/user-attachments/assets/3cc15be0-bdcf-448e-9b4a-88f750f5466e)


**Grafana Dashboard View 1**


![image](https://github.com/user-attachments/assets/481ae73f-cd33-40e7-b108-b28f9f657f1b)


**Grafana Dashboard View 2**


![image](https://github.com/user-attachments/assets/83746db7-737f-4fc9-b636-2920e415a8fb)




