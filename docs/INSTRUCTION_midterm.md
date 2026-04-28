Hi @everyone! Just a heads-up — your Midterm Challenge is coming up next Wednesday, April 1st, 2026! Here's everything you need to know:

:alarm_clock: Time: 10:00 AM – 8:00 PM
:clipboard: Deliverables

Report (3 pts)
Format: PDF (use the provided template)

Competition (7 pts + 1 pt bonus)
Format: Game file in .npy format and your code in a single .zip
The .npy file will be automatically saved in ./data/save when you finish your navigation (Normally by press space).



:bar_chart: Rubric Breakdown

Participation — 1 pt
Achieving the Goal — 2 pts
Position error < 0.1 → 2 pts
Position error 0.1–0.2 → 1 pt
Otherwise → 0.5 pt

Navigation Time — up to 2 pts
Under 1 min → 2 pts
Under 2 min → 1.5 pts
Under 5 min → 0.5 pts

Methodology & Judge Evaluation — up to 3 pts
Completely new solution → 2.0 pts
Modification of baseline → 1.5 pts
Baseline solution → 0.5 pts
+ 1.0 pt bonus for a fully automated solution, we will check your code!

Bonus Points [1 pt] — Can be carried over to the final competition!


:key: Evaluation Platform & API Key
Evaluation Platform: https://ai4ce.github.io/vis_nav_player/
Check your Slack DMs — you should have received a personal API Key to access the Evaluation Platform. Please make sure you've got it!

The platform is there for you to verify your results (see the translation error & navigation time) during development — it's not the submission portal. On the day of the challenge, you'll still need to submit via the Google Form, which requires:

Your exported .npy game file
You code in a single .zip  file
Your report (PDF)
Your team members' information


:world_map: Development Maze & Dry Run
To help things run smoothly, here's how the platform works:

You can validate your solution using your development maze.
After submission, you'll be able to see the maze's BEV (Bird's Eye View) and your trajectory for development maze only.


We've also added a MidTerm Dry Run :tada: — this is specifically to help you test whether you can successfully switch to a new maze. On the day of the challenge, you'll need to switch to a new maze and use new exploration data, so please use the dry run to make sure your pipeline handles this correctly! You can also use this dry run as a mock mid term challenge!