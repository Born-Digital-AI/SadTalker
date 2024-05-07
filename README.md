# Sadtalker UI

## Run locally in Docker:
1. `cp .env-example .env`
2. Fill the necessary env variables
3. `docker-compose up -d --build`

If you want to run the application on a CPU instead of Nvidia GPU, use the Dockerfile_cpu instead and delete the deploy specification in docker-compose.yaml

## Build:
`docker buildx build --platform linux/amd64 -t borndigitalaibot/sadtalker:0.0.1 --push .`

## API Usage
````yaml
swagger: "2.0"
info:
  version: "1.0.0"
  title: "Sadtalker API"
paths:
  /status/{job_id}:
    get:
      summary: "Get the status of a job"
      parameters:
        - in: path
          name: job_id
          required: true
          schema:
            type: string
          description: "The job id"
      responses:
        200:
          description: "Job status result"
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "unknown"
                  result:
                    type: string
                    example: "job return value"
                   
  /generate:
    post:
      summary: "Generate a complete avatar"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/definitions/GenerateRequest'
      responses:
        200:
          description: "ID of a queued job"
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    example: "Generated job id"
                   
definitions:
  GenerateRequest:
    type: "object"
    properties:
      source_image:
        type: "string"
        format: "byte"
        description: "base64 string of the source image"
      bg_image:
        type: "string"
        format: "byte"
        description: "base64 string of the background image"
      email:
        type: "string"
        description: "email where the link to the generated avatar will be sent"
      avatar_name:
        type: "string"
        description: "unique name of avatar to be generated"
      preprocess_type:
        type: "string"
        enum: ["resize", "full", "crop", "extcrop", "extfull"]
      is_still_mode:
        type: "boolean"
        description: "Flag indicating the mode of the video (true = more static video)"
      exp_scale:
        type: "number"
        format: "float"
        description: "The expression scale (determines emotions and movement of the avatar)"
    required:
    - source_image
    - email
    - avatar_name
    - preprocess_type
    - is_still_mode
    - exp_scale
````