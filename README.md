This is the final capstone for my flora/fauna project, which includes: running biodiversity metrics on mythology, emotion analysis on flora/fauna, and topic modeling, among many other tasks. Please enjoy my emoji-laden code.
My database isn't public yet, as I'm still collecting myths (I'm at 6400/7000). I'm going to the Boston Digital Humanities Symposium, and will have the database available online before then!
Note that this capstone is in progress -- I'm hoping to add more detail on this page as the semester continues.

*Updates as of 3/5:*
At the moment I'm hardcoding the plurals of my flora/fauna, as I find there aren't many flora/fauna words in stories in English and regex+hardcode is honestly faster and more effective than NER for this task (though I may train NER on it in the future). I was lemmatizing earlier,  which was a pain, so I'm just adding the plural of each flora/fauna to my CSV files.
I also cleaned logic and code in the utils file --> but this will be much cleaner once lemmatization is removed.
Lemmatizaion of stories is also slooooooow, so having a list of plurals will significantly speed up the program.
