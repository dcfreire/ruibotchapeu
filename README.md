# RuiBot Chapeu

RuiBot is an, *under development*, autonomous snooker referee. Using video frames from only one camera RuiBot should be able to automatically compute an homography such that it ignores everyghing that is not the table. It detects and track the snooker balls and their collisions. Checking which balls were potted, RuiBot should be able to score plays and detect fouls.

## Progress

- [x] [Automatic Homography](#homography)
- [x] Ball Detection
- [x] Ball Tracking
- [x] [Collision Detection](#collision)
- [ ] Pot Detection
- [ ] Score Pots
- [ ] Detect Fouls

## Results

### Homography

![homography](./misc/homography.gif)

### Collision

<p align="center"> 
    <img src="./misc/collision.gif" width="396" height="500"> 
</p>
