import os
import csv
import torch
import datetime

def export(model, test_loader, device, epochs, batch_size, total_loss):
    model.eval()  # Switch to evaluation mode

    results = []
    with torch.no_grad():  # No gradients are needed during evaluation
        for inputs, file_ids in test_loader:  # `file_ids` are the IDs from the dataset
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Convert outputs to a list of predicted values
            predictions = outputs.cpu().numpy().flatten()
            # predictions = int(abs(float(predictions)))   # KEREKITES
            # Combine file IDs and predictions for saving
            for test_id, prediction in zip(file_ids, predictions):
                results.append([test_id, prediction])  # Ensure `test_id` is used as-is, without splitting

    today = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file = f'results/{today}_solution_epoch-{epochs}_batch-{batch_size}_RMSE_min-{round(total_loss, 4)}.csv'

    # Create missing directory if it doesn't exist
    folder_path = os.path.dirname(output_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"A mappa létrehozva: {folder_path}")

    # Write results to the file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Expected'])
        writer.writerows(results)

    print(f"Eredmények kiírva: {output_file}")