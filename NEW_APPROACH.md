First Phase of changes:
- When a user decides to change a concept art (modify/regen/restyle) don't delete the old art, in fact keep the old art at the same location/name.  Instead just add another index to the concept art names (e.g. concept_art_1_0) where the second index is the number of times the image has been regenerated.
- The mesh generation folder name gets the new index, (e.g. hypership_1_0).   Where hypership is name of pipeline, first index is concept art index, 2 index is number of concept art changes

Updates tests and make sure they pass

Second Stage of changes:

I want to make some changes at the API level to make things more generic for developers that might want to develop a tool that only does mesh generation or only for image generation.

There are two different pipelines, at the api level


2D Pipeline
- Everything up until the approval part is the same as the current pipeline.  In other words:
    - Submit an image or image + prompt
   - Can loop through this any number of times
       -  Poll for state
        - Get x number of images + review sheet back from sever
        - Choose a image based on indices to change
    - No more approval API command

It's important that the user of the front end has the ability to receive the images on their local hard drive

3D Pipeline
-  New API command submit image for mesh generation
   - Image can be submitted by 2D pipeline identifier + image indices of 2d pipeline 
     - Server can find the image that should already exist on it's local hard drive
    - Optionally:  Image contents itself can be submitted by uploading an image to the server 
    - Either way, this pipeline is exactly the same as the the end of the original pipeline after image approval:
        - Generate mesh
        - Texture mesh
        - Clean up-mesh
        - Approval of mesh still exists, if rejected then go back to generate mesh phase
- 3d pipelines get their own folder under the pipelines directory, but if they were generated from a 2d pipeline image, their naming scheme can be used to determine which 2d pipeline they came from
     -  If this isn't done already, the server should check for duplicate names when creating 2d/3d pipelines

Update tests and make sure they pass at this point.

Third Stage of changes:
CLI Changes:

Please consider re-writing the cli, and it's associated test entirely, but only if there is a significant amount of changes at this level 

- A new additional feature: you can start a 3d pipeline in the mesh generation stage by submitting an image from local hard drive
   - They provide a name and it becomes a 1st class pipeline
   - It will have it's own pipeline folder, with the same structure (sans concept arts folder)
- No longer a concept of a completed pipeline (so merge the completed/uncompleted pipeline folders in to one)
    - A 2d pipeline can be idle after it's finished generating/changing images
   -  A 3d pipeline can be idle too
    - A user can choose "return to pipeline", in which they can return to an idle 2d pipeline, and are given options to change images, or submit them from mesh generation.  Same for 3d pipelines, they can choose to regenerate a mesh with a different number of polygons, and approve.  If multiple approvals come from the same 3d pipeline that results in multiple exports, add a version number and increase each time the final assets are generated.
    - A user can resubmit the same image from 2d pipelines that already has/had a mesh generation pipeline, just warn the user, give them the option to say nevermind and cancel, and if they do re-submit, just cancel the current 3d pipeline, and overwrite the pipeline folder if it already exists
- When the user approves images for mesh generation, they are using the new submit API to submit them
   - This starts a new 3d pipeline, with the naming scheme mentioned in previous stages

- No more pause/resume/cancel from the CLI.  instead provide the following:
   - Hide 2d/3d pipeline, removes from CLI pipeline list but doesn't destroy pipeline folders
   - Restore hidden pipeline
   - Kill 2d/3d pipeline, remove the pipeline entirely, including destroying the folders containing the generated content (not reversible, so provide warning, and a chance to cancel)


   Answered Questions:

   Stage 2 — significant gaps

2D pipeline end state: "No more approval API command" — but the current 2D pipeline sits in CONCEPT_ART_REVIEW waiting for approval. What state does it move to when the user is done with it? You need an explicit IDLE or DONE status, otherwise the pipeline just hangs. Relatedly: does submitting images to the 3D pipeline change the 2D pipeline's status at all?

- Answer: No, there is no longer a concept of a pipeline being finished because the cli now has an option that says hey i'd like to bring up this pipeline to regenerate or submit meshes from it for a 3d pipeline.

Where do 3D pipeline folders live? Stage 2 says "3D pipelines get their own folder under the pipelines directory." Stage 3 says to merge completed/uncompleted into one folder. These two stages need to agree on the root folder before you write Stage 2 tests. Otherwise Stage 2 tests will be wrong and need rewriting in Stage 3.
My suggestion: adopt the unified pipelines/ directory in Stage 2 already. It's less rework.

- So ultimately there is now just a pipelines folder under it can be 3D pipelines or 2D pipelines.  The 3d pipelines spawn from 2d images will automatically get folder names based off the image indices.  3d pipelines spawned from a custom/outside image get their own name

3D pipeline created directly (no 2D parent) — naming: You say the user provides a name. But what if that name collides with a derived 3D pipeline name (e.g., user names their pipeline hypership_1_0, which would collide with a derived one from hypership CA slot 1 version 0)? Worth a naming convention or namespace separator to distinguish user-named vs. derived pipelines.

- Sure use a naming convention or namespace seperator whatever is easier

"Receive images on local hard drive": The current download endpoint exists but downloads one image at a time. Do you want a bulk download / zip endpoint, or is per-image sufficient?

- Per-image is sufficient for now

Duplicate name checking: Mentioned for both 2D and 3D pipelines but not currently implemented. Easy to add at pipeline creation time, just make sure it's scoped to the right directory.

Stage 3 — gaps

Multiple exports from the same 3D pipeline: You mention a version counter for repeated approvals. This counter needs to live on the 3D pipeline state explicitly, otherwise the export path derivation has no source of truth.

0 That's fair, you can add it to a state file or whatever makes more sense.

Overwriting a 3D pipeline: "Cancel the current 3D pipeline and overwrite the folder." If the pipeline is actively generating, what's the cancellation mechanism? Is it a hard kill (same as the kill command) or a soft stop? The CLI also has a separate kill command that destroys folders — clarify whether overwrite is just an implicit kill+recreate.

- It's a hard kill.

Returning to a 3D idle pipeline: You say the user can "regenerate a mesh with a different number of polygons." Does this reset the 3D pipeline state to GENERATING_MESH and overwrite the previous mesh, or does it create a new export version? These are two different behaviors.

- Have it create a new export version.

Hidden pipelines and folder structure: When hiding a pipeline, do you just set a flag in state (the pipeline folder stays put), or does the folder move somewhere? A state flag is simpler and safer.

- State flag is fine.

The CLI rewrite threshold: "Only if there is a significant amount of changes at this level." Given that pause/resume/cancel are being removed, hide/restore/kill added, return-to-pipeline added, and 3D-pipeline-from-file added — that's almost certainly enough to justify a rewrite. Worth deciding upfront rather than mid-implementation.

- Yeah go ahead and rewrite.  Diffing a file with a tons of changes in git is no fun.

Cross-cutting gap

You'll need a new top-level state model for 3D pipelines (something like Pipeline3DState) alongside the existing PipelineState (2D). The current PipelineState.meshes: list[MeshItem] would be removed from 2D state and become the body of the 3D state. This is the biggest structural change and affects the API, state serialization, and all existing mesh tests.

Understood

Want me to propose a concrete data model and folder layout before coding starts?

 - Yes