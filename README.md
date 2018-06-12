# tensorflow-dataset-api
Examples demonstrating dataset api capabilities.


1. [Basic Placeholders](EP1_Basic_Placeholders.py)
2. [Basic Dataset](EP1_Basic_Dataset.py)
3. [Feedable Dataset](EP1_Feedable_Dataset.py)
4. [Generator Dataset](EP1_Generator_Dataset.py)
5. [Feedable Iterator](EP1_Feedable_Iterator.py)
6. [Feedable Generator Dataset](EP1_Feedable_Generator_Dataset.py)
7. [Feedable Generator Feedable Iterator Dataset](EP1_Feedable_Generator_Feedable_Iterator_Dataset.py)


#### Note:

If facing issues in downloading the MNIST Dataset, use below bash command to update required certificates
```bash
/Applications/Python\ 3.6/Install\ Certificates.command
```


#### Choice of Benchmarks
1. EP1
2. EP3
3. EP5
4. EP11
5. EP12
6. EP13



# Steps
1. Placeholder for Batch Size - For every dataset (Dataset API)
2. Handle for Iterator Switching (Model Runner / Flow API)
3. Dataset Creation - (Dataset API)
4. Initializable Iterator for Dataset - (Dataset API)
5. Feedable iterator - (Iterator Component API)
6. Placeholders with Default - (Inbuffer API)
7. String Handle for iterator (Dataset API)
8. Create Session (Model Runner / Flow API)
8. Fetch handle from running string handle (Model Runner / Flow API)
9. Initialize Iterator (Flow API)
10. Close Session (Model Runner / Flow API)