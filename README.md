# compod_backend

1. Create the Virtual Environment
cd your-project-folder

# Create the virtual environment
python -m venv venv
2. Activate It
venv\Scripts\activate

3. Install Your Dependencies
   pip install -r requirements.txt
4- create .env add aws credentials in env file
   AWS Access Key ID: your_access_key
AWS Secret Access Key: your_secret_key
Default region name: ap-southeast-1
   LOCAL_DEV=true
5.run the backend in local
uvicorn main:app --reload --port 8000 
