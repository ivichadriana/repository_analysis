# Analysis of GitHub repository and project content.

The following repository contains an A.I workflow to generate summaries of GitHub repositorie's code and work over an X amount of time. We use cheaper older models for low-level tasks, and newer costly models for high-level tasks. 

You can change time (for activity) and models used in the [src/full.sh](src/full.sh) script. 

Please note that projects_seeds.csv is created within the pipeline, and if you want to test the pipeline with other or less projects, adjust[src/full.sh](src/full.sh) accoridngly.

Also note that the [src/full.sh](src/full.sh) pipeline removes all documents currently in reports and data folders. Plan accordingly before running.

## 1. First add the 2 needed tokens: one for GitHub and one for OpenAI API.

### - GitHub: 
#### - Go to [GitHub.com](https://github.com/) > Seitings > Developer settings > Personal Access Tokens > Tokens (classic) > Generate a token, select repo box and read:org box. You need to have nih-cfde oganization access.
### - OpenAI API: 
#### - Go to [OpenAI API](https://openai.com/api/pricing/), generate token to use 
### - Add tokens to the .env file (.env_example can serve as template)

## 2. Run the full pipeline: 

```bash
docker build -t cfde-pipeline --no-cache .

docker run --rm --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/reports_pdf:/app/reports_pdf" \
  cfde-pipeline
  ```



