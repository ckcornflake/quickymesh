# Example prompts

A small collection of descriptions that have produced good results as the
`description` field in `[n] Start a new 2D pipeline`. Use them as-is or as a
starting point for your own prompts.

quickymesh automatically appends a background/lighting suffix
(`"3/4 isometric view, three quarters lighting, plain contrasting background,
1:1 ratio"` by default) to every description — you don't need to include
framing or lighting instructions yourself. Focus on the subject.

## What works well

Single objects with a clear silhouette, roughly symmetric, with recognizable
features. Trellis struggles with very thin geometry (rope, chains, wires), deep
concave holes, and objects that are mostly transparent.

## Props and weapons

- `an ornate medieval sword with a jewel-encrusted pommel and a leather-wrapped grip`
- `a wooden treasure chest reinforced with rusted iron bands`
- `a steampunk pocket watch with exposed gears and a brass chain`
- `a wizard's staff topped with a glowing blue crystal wrapped in twisted vines`

## Creatures

- `a red dragon curled around a pile of gold coins, wings folded`
- `a small fluffy cat wearing a tiny knight's helmet`
- `a mushroom-shaped forest spirit with glowing yellow eyes and a mossy cap`

## Vehicles and structures

- `a small wooden fishing boat with a patched sail`
- `a low-poly stone well with a wooden roof and a rope bucket`
- `a retro-futuristic rocket ship with fins, portholes, and checkered decals`

## Environment props

- `a gnarled oak tree stump with mushrooms growing on the side`
- `a medieval wooden barrel bound with iron hoops`
- `a rune-inscribed standing stone covered in moss`

## Tips

- Keep descriptions to one or two sentences. Longer prompts tend to produce
  busier images with more concept art variance.
- Mention materials explicitly (`wooden`, `stone`, `brass`, `leather`) —
  Trellis textures come out noticeably better when the input image has clear
  material cues.
- If you want symmetry, say so in the prompt *and* pick a symmetry axis at the
  prompt step (`x-` is a good default for creatures and props).
- If the first batch of concept arts isn't great, use `regenerate all` with a
  refined description rather than approving a mediocre image — mesh quality
  tracks image quality closely.
