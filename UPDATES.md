- You haven't yet implemented the ability to cancel/pause/resume/recover a pipeline from the idle prompt
    - Pausing the pipeline means that its current state is saved, and when resumed, attempt the next uncompleted step of the pipeline
    - Canceling the pipeline means it won't show up on the pipeline list anymore
    - Recover pipeline.  It should be relatively easy to detect what state a pipeline is in based on it's state file of what has been generated under it's uncompleted_pipeline folder
        - It shouldn't matter whether the pipeline was canceled or it's db info got clobbered

- Have you implemented symmetrize option yet?
    - I see the models aren't perfectly symmetrical, but I also haven't seen any options in my prompts to change symmetrize settingsn
    - If symmetrize option is on for the pipeline, and is set to auto, a blender script will symetrize along the most likely axis where the model should be symmetric against
    - Otherwise the symmetrize function will symmetrize based on the axis and side the user specifies
    - This prompt should appear the start of the pipeline, but can be changed until the mesh is generated

- I also want the ability to modify a pipeline because it can be stuck in the concept_art_generation phase due to a too big of a prompt
    - [e]  Edit pipeline
        - Ask if they want to start with image 
        - Provide current model description prompt and a chance the user to change it
        - Provide the number of polygons and a chance to edit it
        - Provide current symmetry setting, and give the chance for user to change it

- Do you automatically retry when a gemini api request fails?  If you do make sure you double the delay amount before you try again each time (also inform the user you are doing so)
w
- I want this feature added
    - When user wants to create a new pipeline, they should be asked for an image location relative to pipeline root.  When prompted for this image say something along the lines, "Do you want to base the concept art of a previously made image (leave blank to create concept art from prompts alone)".  Hopefully you come up with something more concies.
    - If a user does choose the image, prompt them for the image prompt by asking "how do you want to change this image?" then it should be treated similar to how they can modify an image: they send the API the image and a prompt the modifies the image

- For initial prompts (sans image and initial base image), automatically add the following words to the prompt "3/4 isometric view, three quarters lighting, plain contrasting background, 1:1 ratio"
    - This should be added as a suffix for every use prompt
    - This should be configurable through defaults.yaml
    - When the user enters their prompts inform them that the suffix will be added and what is in the suffix and why (because trellis requires these qualities in order to see the object as 3d)

- Can you make the background of the 3d preview a little bit darker (can be a light gray, just not white).  And maybe center it in the webpage

- In this previous pipeline, (location mentioned in docs.md), I think we edited the UVs or the textures slightly so that the models looked smoother
    - Currently, at certain angles, I can see each individual polygon face becuase of how different the light bounces off
    - I think we did something with blender to make the normals look smoother, but I could be mistaken.  Is there a technqiue whether it's a comfyui node or a blender function that can do this?

- Other than canceling it there should be a way to deal with a pipeline stuck like this (perhaps a retry, which is like a pause/resume under the hood):

```
=== Pipelines ===
  spaceship_one: initializing  [1 FAILED]
    ! concept_art_generate: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'Deadline expired before operation could complete.', 'status': 'UNAVAILABLE'}}

Enter number or text: w
Watch mode active.  Updates will appear below as pipelines progress.
Press 'q' to return to the menu.
[21:35:59] spaceship_one: initializing [FAILED: concept_art_generate]
[21:35:59]   ! concept_art_generate error: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'Deadline expired before operation could complete.', 'status': 'UNAVAILABLE'}}

```

- The final pipeline got seemingly stuck after generating the final game assets after the user approved two pipelines.  See below:

```
2026-04-01 21:10:46,799 INFO src.agent.worker_threads: Task 5 done

[21:10:46] !!! APPROVAL NEEDED: spaceship_two !!!

Mesh 'spaceship_two_1' is ready for review.
Actions:
  approve <asset_name> [format]  — approve and name the asset
  reject                          — reject this mesh (pipeline continues)
  cancel                          — cancel the pipeline
  quit                            — exit the program
Enter action
> approve protoss_1 glb
Approved 'protoss_1' (glb). Will be exported after all reviews.

Mesh 'spaceship_two_3' is ready for review.
Actions:
  approve <asset_name> [format]  — approve and name the asset
  reject                          — reject this mesh (pipeline continues)
  cancel                          — cancel the pipeline
  quit                            — exit the program
Enter action
> approve protoss_2 glb
Approved 'protoss_2' (glb). Will be exported after all reviews.
[spaceship_two] Meshes exported to C:\Users\jmkel\claude_projects\quickymesh\pipeline_root\final_game_ready_assets.
```

At this point I had to press q to unfreeze the CLI, but I didn't realize I was still in watch mode.  Either inform me I'm in watch mode if you return to it after an approval prompt, or return to the idle prompt automatically.