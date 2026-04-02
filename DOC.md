You will create a 3d modeling pipeline that uses Gemini Flash API to generate and modify images, and a Trellis2 comfyui workflow to generate and texture the resulting models.  Multiple instances of the pipeline below should be supported.

The pipeline will have the following steps:

1.  Makes a request to a program/agent to generate a new set of models under a given name.  This starts a new pipeline under a new directory under the repo
    - Optionally, at this time the user can also supply the human language description of the model.  You will suffix this request with "A plain white background" so that whatever the user requests, ends up with a white background.  Also inform the user of the addtional suffix appended to the requested, and why it needs to be there.
    - If the user doesn't supply a description of the image, the program will ask them for one at this time
    - Optionally the user can supply an image so that they can tell gemini to make something similar through the description prompt
2.  The user's description will be submitted to the GEMINI API under the {insert_flash model here} model X number of times
    - The image should be 1:1 ratio, so it will be easy to change it to 1024x1024 if needed, (change it by pasting it onto background, don't rescale it)
    - For each time it should generate a different image with a different concept art, if you have to add "make a different version of this request" to the prompt than do so, but it seems like just submitting the same prompt over and over seems to work
    - X is configurable through a defaults file
3.  When the images are done being generated, they should be all saved into a folder under the pipeline folder
    - You will concantenate the images into a review sheet where each image is labelled and has a size of 256x256 (this size should be configurable), and the columns/row # should modified to make the review sheet image as close to a square as possible
    - The user has the following choices:
        - Approve a list of concept arts that are good enough to be used to generate models.  Once the user chooses this option, the pipeline will move on to the next phase automatically, so inform the user of this
        - Regenerate a subset of the images by specifying their index (this just uses the same prompt but submits another api request)
        - Modify an image through geminis api (you can submit an image, and make human language requests like "remove the wheels from this truck").  Before they choose this option, make sure the user knows that it will replace the origional concept art
        - Cancel the pipeline, so that the program stop's trying to bother the user or go on to the next phase
        - Quit the program
4.  The each selected image now will use trellis workflow to generate the mesh.  I will provide the comfyui workflow that I used for a previous pipeline project.
    - This workflow works best with 1024x1024 images, so hopefully the original images you saved will be this format, otherwise you will have re-size the image (without distorting the original art, so just add more background)
5.  Then the image will be textured by the texturing trellis workflow.
    - At this phase you will bake the anything that needs to be baked and generated unwraped uvs of the textures invovled
    - There might be a clean-up phase to adjust the normals here, because they aren't quit right when they come out of trellis
6.  Once textured mesh is generated, a program will open up blender (through python), and take screenshots of the mesh
    - Front, back, bottom, top, perspective 1, perspective 2 screenshots will be taken and concatenated into a single review sheet
    - Also you will use TriMesh python library to generate .html preview where I can rotate the model around in a browser.
7. The user will then be prompted for final approval of the mesh.
    - They must approve and give a name for the final asset
    - At this point they have the option to change the format of the final output (but default is .glb)
    - IF they approve, symmetrize the model if user wants it, and export the models as game ready assets into a separate folder higher up in the hierarchy
    - You should know which axis to symetrize on, but the user should be able to specify which axis to symmetrize on (and which side), in case you get it wrong
        - So the options should be auto, x-, x+ y-,y+, z-, z+


Note: Trellis worfklows already created.  In:  comfyui_workflows directory

Requirements:
- Defaults file
    -  number of concept art images generated in the first phase (4)
    -  num polys if the user doesn't specify it (8000)
    -  size of the concatenated images in the review sheet (256x256)
    -  size of the 3d preview html thing (512x512)
    - Format of final mesh (.glb) options (whatever other formats blender can xport)
- The idea of "prompting" the user should be abstracted
    - The first version of the pipeline will be a CLI program, but we will also support prompting the user through emails as well, also an agent should also be able to make requests to the pipeline
    - This program/agent should be capable of handling multiple pipelines at a time
    - If there is nothing waiting for user approval there should be an idle prompt. At idle prompt phase
        - The user will be shown the most up to date status of the workers and pipelines
        - The user can request to refresh to get a more recent update of the status of workers/pipelines
        - The user can request to pause and resume a pipeline
        - THe user can request to kill a pipeline, the main difference from pause is that it won't show up in the list of pipelines in the pipeline status output
        - User can start a new pipeline - if they only specifiy a name then prompt them with a list of questions that is required for the pipeline: num_polys, symmetrize, description
    - The order of prompt priority should be: new pipeline questions (if they didn't enter num_polys/symmetrize/description at idle prompt) -> mesh approval -> concept art approval -> idle
        - Any prompt that is higher priority will trigger and if it's an approval than the next highest priority prompt appears
- Since mesh generation takes a lot longer than concept art generation, and we want the user to not have to wait on anything, there should be multiple processes
    - concept art worker that makes gemini api requests as soon as it receives a concept art generation/modification request
    - a trellis/comfyui worker that submits an image to the trellis worfklow as soon as the user approves it for mesh generation
    - a screenshot worker that is reponsible for opening up blender taking and screenshots, concantenating the screenshots into a review sheet, this worker can also be responsible for concantenating review sheet for the concept art phase as well.  It will have to know when the pipeline is done generating screenshots to beging the image combination.
    - The main pipeline agent that is responsible for maintaining the worker threads/processes
- Each pipeline has the following settings
    - Number of polygons for the mesh
    - Whether the mesh will be symmetrized after it's approved
- The state of the pipeline should be saved so it can be recovered when the pipeline process starts up
    - The prompts used to generate and modify the ship must also be saved in the state file
- Each time the user is prompted during a pipeline, as long as it's before mesh generation, they have the option to change the target polygon count of the mesh that will be generated
    - This choice should be saved in the state file
- For convenience the pipeline should know where the comfyui install is so it can start up the comfyui instance when it can't connect to it and verifies it's not running
- Only one pipeline agent (who is capable of running multiple pielines) need to interface with the workers

Development Plan:
- Unless you think there is a reason for another programming language, all code will be written in python
- You must use the test driven development to develop the code, so use pytest for unit tests
    - This means that you should likely abstract the prompting enough so that another non-human agent can make test prompts and requests and verify the output
    - Make sure you define what success looks like to an agent without eyes.  If you need a human to approve that some aspect of your code is working, please don't be afraid to ask.
        - For example, you will need a human to verifiy that the blender screen shots and concatenation step looks correct.
        - Otherwise if it's to ensure that the pipeline simply produced an image, or a mesh, etc. and agent should be able to verifiy this
        - Once the full parelell-pipeline approach is implement we will need to test a bunch different combinations of prompts, this will likely require both human and AI testing
- You should develop iteravely
    - Make sure each phase/step of the pipeline runs individually, and successfully first before going on to the next phase of the pipeline
        - If you need help from the human here to try things, that's okay
    - Don't implement all the workers and everything all at once, make sure each worker is running without issues before putting the entire parallel-pipeline agent together
    - Don't worry about the email agent until the cli agent is confirmed working
- I have blender isntalled under C:\Program Files\Blender Foundation\Blender 4.5
- My trellis comfyui workflow is accessible through port 8188
    - The protable comfyui instance is in  C:\Users\jmkel\Downloads\ComfyUI_windows_portable_nvidia_cu128\ComfyUI_windows_portable
- Eventually we will dockerize this workflow so keep that ind mind
- Use my old pipeline to see how to implement things at the lowest atomic level (e.g. like how do I talk to blender or trellis, or how do i make good screenshots, or how do I concantenate them)
    - Location: C:\Users\jmkel\claude_projects\ship_pipeline_previous
    - DO NOT BASE THE STRUCTURE of the new code from this pipeline, only read it to learn how to do some of the lower level things, (i.e at most you might copy individual functions, but do copy entire files from here)
- Produce good documenation about how to run/test the pipeline
- Produce any other documentation about the status of the project that future agents might need to resume the pipeline.


Additional Architectural Components:


- A Shared Message Broker/Queue: Since you have multiple workers (Gemini, ComfyUI, Blender), don't rely on simple global variables. Use a lightweight, free broker like Redis (or even a simple SQLite-based queue) to manage tasks. This ensures that if the Python script crashes, the Trellis worker doesn't lose its "place" in the 5-minute generation process.
- The "VRAM Arbiter": TRELLIS and ComfyUI are VRAM hungry. If your worker tries to run a Trellis mesh generation while Blender is open and another pipeline is generating textures, you will hit Out of Memory (OOM) errors. You need a simple "Resource Manager" that ensures only one GPU-heavy task runs at a time.
- Asset Versioning: Since users can "Regenerate" or "Modify," the folder structure needs to support versioning (e.g., ship_01/v001/concept_art/). This prevents the UI from showing a cached version of an old image.




Development Efficiency (The "Claude Code" Strategy) Suggestions:


Mock the Workers: For TDD, have Claude create "MockWorkers" that return dummy images/meshes instantly. This allows you to test the complex Pipeline State Machine and Idle Prompt logic without waiting 5 minutes for a Trellis model to bake every time you run a test.

Pydantic for State: Use Pydantic to define your Pipeline State. It handles the JSON serialization/deserialization to your state file automatically and provides strict type-checking, which will save you dozens of hours of debugging "state recovery" issues.



The asset structure of the pipelines should look like this


- final_game_ready_assets
    - final_chosen_name
        - .glb, and whatever the game or 3d printer needs go here
        - A text file that contains the location of the original concept art, and the location of its completed pipeline folder
- completed pipelines
    - pipeline_name1
        - concept_art
            - concept_art1.png
            - concept_art2.png
            - concept_art3.png
            - concept_art4.png
            - reviewsheet.png
        - piepline_name1_1   < if 1 & 2 are chosen to made into meshes
            - meshes
                - initial trellis mesh
                - cleaned up trellis mesh
                - textured mesh
            - unwrapped_textures
                - ao
                - normal
                - metallic,etc, if they exist
            - screenshots
                - all the different views go here
                - review sheet
            - 3d previewer html
        - pipeline_name1_2
                - meshes
                - initial trellis mesh
                - cleaned up trellis mesh
                - textured mesh
            - unwrapped_textures
                - ao
                - normal
                - metallic,etc, if they exist
            - screenshots
                - all the different views go here
                - review sheet
            - 3d previewer html
- uncompleted pipelines
    - pipeline__other_name1
        - etc, etc
