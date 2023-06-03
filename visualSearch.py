import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
import time

def train_model(model,train_dataset,valid_dataset,device,optimizer, 
                loss,batch_size=128,num_epochs=20,patience=10,
                output_filename='trained-network.pt'):
    
    # Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    # Set model to training mode
    model.train()
    
    # Initialize lists to contain losses and accuracies
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    
    # Initialize patience counter
    patience_counter = 0

    print('Training has started!')
    
    # Training and validation loop
    for epoch in range(num_epochs):
        
        # Record start time
        start_time = time.time()
        
        # Intitialize 
        count = 0
        total_loss = 0.0
        correct = 0
        
        # Train over batches 
        for inputs, targets in train_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss_value = loss(outputs, targets)
            
            # Accumulate the number of processed samples
            count += inputs.shape[0]
            
            # Accumulate the total loss
            total_loss += inputs.shape[0] * loss_value.item()
            
            # Compute total accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            
            # Backpropagate the error to change the model weights
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
        # Append the last train loss and accuracy to the list
        train_losses.append(total_loss/count)
        train_accuracies.append(correct/count)
        
        #________________________________________________________________
        
        # Intitialize 
        count = 0
        total_loss = 0.0
        correct = 0
        
        # Validate over batches
        for inputs, targets in valid_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss_value = loss(outputs, targets)
            
            # Accumulate the number of processed samples
            count += inputs.shape[0]
            
            # Accumulate the total loss
            total_loss += inputs.shape[0] * loss_value.item()
            
            # Compute total accuracy
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            
        # Append the last validation loss and accuracy to the list
        valid_losses.append(total_loss/count)
        valid_accuracies.append(correct/count)
        
        #________________________________________________________________
        
        # Record finish time
        finish_time = time.time()
        # Calculate elapsed time
        elapsed = finish_time - start_time
        
        print(f'Epoch {epoch+1} done! ({epoch+1}/{num_epochs}). Elapsed time: {int(elapsed)} seconds')
        print(f'     Loss: {train_losses[-1]} - Validation Loss: {valid_losses[-1]}')
        print(f'     Accuracy: {train_accuracies[-1]} - Validation accuracy: {valid_accuracies[-1]}')
        
        #________________________________________________________________
       
        # Save the best model
        if epoch == 0:
            torch.save(model, output_filename)
            print('     Model saved!')
        elif valid_losses[-1] < min(valid_losses[:-1]):
            torch.save(model, output_filename)
            print('     Best model so far, saved!')
            patience_counter = 0
            
        # Increase patience counter
        patience_counter += 1
        
        # Break out of the loop if the val. loss has not improved in the number of cycles
        # specified by the patience parameter 
        if patience_counter == patience:
            print(f'     The validation loss has not improved in {patience} epochs!')
            break
            
        #________________________________________________________________

    print('Finished!')
        
    return train_losses, train_accuracies, valid_losses, valid_accuracies

def create_embedding_space(train_loader,model):

    embedding_space = []
    labels = []
    with torch.no_grad():
        for inputs, label in train_loader: 
            labels.append(int(label))
            outputs = model(inputs)
            embedding_space.extend(outputs.tolist())   

    return embedding_space, labels   

def create_embedding_test(test_loader,model):

    embeddign_test = []
    with torch.no_grad():
        for inputs, _ in test_loader: 
            outputs = model(inputs)
            embeddign_test.extend(outputs.tolist())

    return embeddign_test

def create_embedding_space_and_test(train_loader,test_loader,model):

    embedding_space, labels = create_embedding_space(train_loader,model)
    np.array(embedding_space)

    embeddign_test = create_embedding_test(test_loader,model)

    embedding_space = np.array(embedding_space)
    embeddign_test = np.array(embeddign_test)   

    return embedding_space, labels, embeddign_test

def create_embeding_PCA(embedding_space,labels):

    # Create a PCA object
    pca = PCA(n_components=2)

    # Fit the PCA model to the data
    pca.fit(embedding_space)

    # Transform the data to the new space
    embedding_space_transformed = pca.transform(embedding_space)

    # Get the unique labels
    unique_labels = np.unique(labels)
    # Add 1 to labels to correspond to class name
    unique_labels = np.add(unique_labels, 1)

    # Plot the transformed data and get the scatter object
    scatter = plt.scatter(embedding_space_transformed[:, 0], embedding_space_transformed[:, 1], c=labels, cmap='gist_ncar')

    # Add legend
    handles, _ = scatter.legend_elements(num=unique_labels.size)
    plt.legend(handles, unique_labels, loc="upper right", bbox_to_anchor=(1.2, 1.0), title="Classes")

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

def calculate_mAP_k(actual, predicted, k):

    relevant_positions = []  
    # Store the positions where actual label is predicted
    for i, label in enumerate(predicted):
        if label == actual:
            relevant_positions.append(i)

    # Calculate the number of relevant positions
    num_relevant = len(relevant_positions)
    if num_relevant == 0:
        return 0

    # Calculate AP
    sum_precision = 0.0
    num_correct = 0
    for i, label in enumerate(predicted[:k]):
        if i in relevant_positions:
            num_correct += 1
            precision = num_correct / (i + 1)  # compute precision at position i
            sum_precision += precision

    # Calculate mAP@K
    mAP_k = sum_precision / num_relevant

    return mAP_k

def obtain_k_nearest_neighbors(image_index,n_neighbors,embedding_test,embedding_space,
                               test_loader_show,train_loader_show):

    # Create a new point
    new_point = np.array(embedding_test[image_index]).reshape(1, -1)

    # Create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding_space)

    # Find the k-nearest neighbors to the new point
    distances, indices = nbrs.kneighbors(new_point)

    # Sort the indices based on the corresponding distances
    indices = [i for _, i in sorted(zip(distances[0], indices[0]))]

    # Load test image from test_loader_show
    for i, (image, label) in enumerate(test_loader_show):
        if i == image_index:
            test_image = image.squeeze().numpy()
            test_label = int(label[0])

    # Load train images from train_loader_show
    train_images = []
    train_labels = []
    for index in indices:
        for i, (image, label) in enumerate(train_loader_show):
            if i == index:
                train_images.append(image.squeeze().numpy())
                train_labels.append(int(label[0]))

    # Plot the test images
    fig, axs = plt.subplots(1, n_neighbors+1, figsize=(10, 4))
    # Remove the axes
    for ax in axs:
        ax.axis('off')
    axs[0].imshow(np.transpose(test_image, (1, 2, 0)), cmap='gray')
    axs[0].set_title(f'Input image (class {test_label+1})', fontsize=8, fontweight='bold')
    for i, image in enumerate(train_images):
        axs[i+1].imshow(np.transpose(image, (1, 2, 0)), cmap='gray')
        axs[i+1].set_title(f'Class {train_labels[i]+1}', fontsize=8)
    plt.show()
    
    # Calculate mAP@k
    mAP_k = calculate_mAP_k(test_label,train_labels,n_neighbors)
    print(f'The mAP@k is {mAP_k}')

def predict_classes(image1_index,image2_index,n_neighbors,embeddign_test,embedding_space,
                    test_loader_show,train_loader_show):
    
    # Combine the indices into a list
    test_indices = [image1_index, image2_index]

    # Create a new point
    new_point1 = np.array(embeddign_test[test_indices[0]]).reshape(1, -1)
    new_point2 = np.array(embeddign_test[test_indices[1]]).reshape(1, -1)

    predicted_class1 = []
    predicted_class2 = []
    for n_neighbors_try in range(n_neighbors, 0, -1):
        # Create a NearestNeighbors object
        nbrs = NearestNeighbors(n_neighbors=n_neighbors_try, algorithm='auto').fit(embedding_space)

        # If there is a tie do
        if len(predicted_class1) != 1:
            # Find the k-nearest neighbors to the new point
            distances1, indices1 = nbrs.kneighbors(new_point1)
            # Get the indices of the k-nearest neighbors
            indices1 = indices1.tolist()
            indices1 = indices1[0]
            # Load train labels from train_loader_show
            train_labels1 = []
            for i, (_, label) in enumerate(train_loader_show):
                if i in indices1:
                    train_labels1.append(int(label[0]))
            # Obtain the predicted class based on the most common class of nearest neighbors
            count = np.bincount(train_labels1)
            predicted_class1 = np.where(count == count.max())[0]

        # If there is a tie do
        if len(predicted_class2) != 1:
            # Find the k-nearest neighbors to the new point
            distances2, indices2 = nbrs.kneighbors(new_point2)
            # Get the indices of the k-nearest neighbors
            indices2 = indices2.tolist()
            indices2 = indices2[0]
            # Load train labels from train_loader_show
            train_labels2 = []
            for i, (_, label) in enumerate(train_loader_show):
                if i in indices2:
                    train_labels2.append(int(label[0]))
            # Obtain the predicted class based on the most common class of nearest neighbors
            count = np.bincount(train_labels2)
            predicted_class2 = np.where(count == count.max())[0]

        # Break out of the loop when a final prediction for both images has been made
        if len(predicted_class1) == 1 and len(predicted_class2) == 1:
            break

    # Load test images from test_loader_show
    test_images = []
    test_labels = []
    for i, (image, label) in enumerate(test_loader_show):
        if i in test_indices:
            test_images.append(image.squeeze().numpy())
            test_labels.append(int(label[0]))

    # Plot the test images
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Remove the axes
    for ax in axs:
        ax.axis('off')
    for i, image in enumerate(test_images):
        axs[i].imshow(np.transpose(image, (1, 2, 0)), cmap='gray')
        axs[i].set_title(f'Ground truth class {test_labels[i]+1}', fontsize=10)
    plt.show()

    # Print prediction
    if predicted_class1[0] == predicted_class2[0]:
        print(f'Both images are predicted to be of class {predicted_class1[0]+1}!')
    else:
        print('Images are predicted to be of different classes!')
        print(f'The first image is predicted to be of class {predicted_class1[0]+1} and the second of class {predicted_class2[0]+1}')

def train_with_miner(model,train_dataset,valid_dataset,device,optimizer,
                             miner=miners.PairMarginMiner(),loss=losses.ContrastiveLoss(),batch_size=24,
                             num_epochs=20,patience=10,output_filename='trained-resnet-contrastative.pt'):

    # Load data
    train_loader = DataLoader(train_dataset,batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset,batch_size=batch_size)
    
    # Set model to training mode
    model.train()
    
    # Initialize lists to contain losses and accuracies
    train_losses = []
    valid_losses = []
    
    # Initialize patience counter
    patience_counter = 0

    print('Training has started!')
    
    # Training and validation loop
    for epoch in range(num_epochs):
        
        # Record start time
        start_time = time.time()
        
        # Intitialize 
        count = 0
        total_loss = 0.0
        
        # Train over batches 
        for inputs, targets in train_loader:

            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Use the miner to obtain anchor, positive, and negative samples
            miner_output = miner(outputs, targets)
            
            # Compute the loss with the selected samples
            loss_value = loss(outputs, targets, miner_output)
            
            # Accumulate the number of processed samples
            count += inputs.shape[0]
            
            # Accumulate the total loss
            total_loss += inputs.shape[0] * loss_value.item()
            
            # Backpropagate the error to change the model weights
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
        # Append the last train loss and accuracy to the list
        train_losses.append(total_loss/count)
        
        #________________________________________________________________
        
        # Intitialize 
        count = 0
        total_loss = 0.0
        
        # Validate over batches
        for inputs, targets in valid_loader:
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Use the miner to obtain anchor, positive, and negative samples
            miner_output = miner(outputs, targets)
            
            # Compute the loss with the selected samples
            loss_value = loss(outputs, targets, miner_output)
            
            # Accumulate the number of processed samples
            count += inputs.shape[0]
            
            # Accumulate the total loss
            total_loss += inputs.shape[0] * loss_value.item()
            
        # Append the last validation loss and accuracy to the list
        valid_losses.append(total_loss/count)
        
        #________________________________________________________________
        
        # Record finish time
        finish_time = time.time()
        # Calculate elapsed time
        elapsed = finish_time - start_time
        
        print(f'Epoch {epoch+1} done! ({epoch+1}/{num_epochs}). Elapsed time: {int(elapsed)} seconds')
        print(f'     Loss: {train_losses[-1]} - Validation Loss: {valid_losses[-1]}')
        
        #________________________________________________________________
       
        # Save the best model
        if epoch == 0:
            torch.save(model, output_filename)
            print('     Model saved!')
        elif valid_losses[-1] < min(valid_losses[:-1]):
            torch.save(model, output_filename)
            print('     Best model so far, saved!')
            patience_counter = 0
            
        # Increase patience counter
        patience_counter += 1
        
        # Break out of the loop if the val. loss has not improved in the number of cycles
        # specified by the patience parameter 
        if patience_counter == patience:
            print(f'     The validation loss has not improved in {patience} epochs!')
            break
            
        #________________________________________________________________

    print('Finished!')
        
    return train_losses, valid_losses

def evaluate_embedding(n_neighbors, embedding_test, embedding_space, test_loader, train_loader):

    # Create a NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding_space)

    # Preload test and train labels
    test_labels = [int(label[0]) for _, label in test_loader]
    train_labels = [int(label[0]) for _, label in train_loader]

    # mAP@K for the whole test dataset
    mAP_k_dataset = []

    for image_index, point in enumerate(embedding_test):
        
        # Create a new point
        new_point = np.array(point).reshape(1, -1)

        # Find the k-nearest neighbors to the new point
        distances, indices = nbrs.kneighbors(new_point)

        # Sort the indices based on the corresponding distances
        indices = [i for _, i in sorted(zip(distances[0], indices[0]))]

        # Load test label from the preloaded list
        test_label = test_labels[image_index]

        # Load train labels from the preloaded list
        train_labels_batch = [train_labels[index] for index in indices]

        # Calculate mAP@k
        mAP_k = calculate_mAP_k(test_label, train_labels_batch, n_neighbors)
        mAP_k_dataset.append(mAP_k)

    mAP_k_dataset = sum(mAP_k_dataset) / len(mAP_k_dataset)

    return mAP_k_dataset


def evaluate_embedding2(n_neighbors,embedding_test,embedding_space,test_loader,train_loader):
    
    # mAP@K for the whole test dataset
    mAP_k_dataset = []

    for image_index in range(len(embedding_test)):

        # Create a new point
        new_point = np.array(embedding_test[image_index]).reshape(1, -1)

        # Create a NearestNeighbors object
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embedding_space)

        # Find the k-nearest neighbors to the new point
        distances, indices = nbrs.kneighbors(new_point)

        # Sort the indices based on the corresponding distances
        indices = [i for _, i in sorted(zip(distances[0], indices[0]))]

        # Load test image from test_loader_show
        for i, (_, label) in enumerate(test_loader):
            if i == image_index:
                test_label = int(label[0])

        # Load train images from train_loader_show
        train_labels = []
        for index in indices:
            for i, (_, label) in enumerate(train_loader):
                if i == index:
                    train_labels.append(int(label[0]))
    
        # Calculate mAP@k
        mAP_k = calculate_mAP_k(test_label,train_labels,n_neighbors)
        mAP_k_dataset.append(mAP_k)

    mAP_k_dataset = sum(mAP_k_dataset) / len(mAP_k_dataset)

    return mAP_k_dataset