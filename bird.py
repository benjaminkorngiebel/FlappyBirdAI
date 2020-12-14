import pygame
import neat
import time
import random
import os
pygame.font.init()

kWindowWidth = 500
kWindowHeight = 800

kGen = 0

birdImages = [pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird1.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird2.png"))), pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bird3.png")))]
pipeImage = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "pipe.png")))
baseImage = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "base.png")))
backgroundImage = pygame.transform.scale2x(pygame.image.load(os.path.join("images", "bg.png")))

statFont = pygame.font.SysFont("comicsans", 50)

class Bird:
    images = birdImages
    maxRotation = 25
    rotationVelocity = 20
    animationTime = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tickCount = 0
        self.velocity = 0
        self.height = self.y
        self.imageCount = 0
        self.image = self.images[0]

    def jump(self):
        self.velocity = -10.5
        self.tickCount = 0
        self.height = self.y

    def move(self):
        self.tickCount += 1

        d = (self.velocity * self.tickCount) + (1.5 * self.tickCount**2)

        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.maxRotation:
                self.tilt = self.maxRotation
        else:
            if self.tilt > -90:
                self.tilt -= self.rotationVelocity

    def draw(self, win):
        self.imageCount += 1

        if self.imageCount < self.animationTime:
            self.image = self.images[0]
        elif self.imageCount < self.animationTime*2:
            self.image = self.images[1]
        elif self.imageCount < self.animationTime*3:
            self.image = self.images[2]
        elif self.imageCount < self.animationTime*4:
            self.image = self.images[1]
        elif self.imageCount < self.animationTime*4 + 1:
            self.image = self.images[0]
            self.imageCount = 0

        if self.tilt <= -80:
            self.image = self.images[1]
            self.imageCount = self.animationTime*2

        rotatedImage = pygame.transform.rotate(self.image, self.tilt)
        newRect = rotatedImage.get_rect(center=self.image.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotatedImage, newRect.topleft)

    def getMask(self):
        return pygame.mask.from_surface(self.image)

class Pipe:
    kGap = 200
    kVelocity = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.topPipe = pygame.transform.flip(pipeImage, False, True)
        self.bottomPipe = pipeImage

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.topPipe.get_height()
        self.bottom = self.height + self.kGap

    def move(self):
        self.x -= self.kVelocity

    def draw(self, win):
        win.blit(self.topPipe, (self.x, self.top))
        win.blit(self.bottomPipe, (self.x, self.bottom))

    def collide(self, bird):
        birdMask = bird.getMask()
        topMask = pygame.mask.from_surface(self.topPipe)
        bottomMask = pygame.mask.from_surface(self.bottomPipe)

        topOffset = (self.x - bird.x, self.top - round(bird.y))
        bottomOffset = (self.x - bird.x, self.bottom - round(bird.y))

        bPoint = birdMask.overlap(bottomMask, bottomOffset)
        tPoint = birdMask.overlap(topMask, topOffset)

        if bPoint or tPoint:
            return True
        return False

class Base:
    kVelocity = 5
    kWidth = baseImage.get_width()
    image = baseImage

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.kWidth

    def move(self):
        self.x1 -= self.kVelocity
        self.x2 -= self.kVelocity

        if self.x1 + self.kWidth < 0:
            self.x1 = self.x2 + self.kWidth

        if self.x2 + self.kWidth < 0:
            self.x2 = self.x1 + self.kWidth

    def draw(self, win):
        win.blit(self.image, (self.x1, self.y))
        win.blit(self.image, (self.x2, self.y))

def draw_window(win, birds, pipes, base, score, gen):
    win.blit(backgroundImage, (0,0))

    for pipe in pipes:
        pipe.draw(win)

    text = statFont.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (kWindowWidth - 10 - text.get_width(), 10))

    text = statFont.render("Gen: " + str(gen), 1, (255, 255, 255))
    win.blit(text, (10, 10))

    base.draw(win)

    for bird in birds:
        bird.draw(win)

    pygame.display.update()

def main(genomes, config):
    global kGen
    kGen += 1
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)

    pipes = [Pipe(600)]
    base = Base(730)
    win = pygame.display.set_mode((kWindowWidth, kWindowHeight))
    clock = pygame.time.Clock()
    score = 0

    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipeInd = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].topPipe.get_width():
                pipeInd = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipeInd].height), abs(bird.y - pipes[pipeInd].bottom)))

            if output[0] > 0.5:
                bird.jump()

        rem = []
        addPipe = False
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    addPipe = True

            if pipe.x + pipe.topPipe.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if addPipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem:
            pipes.remove(r)
        for x, bird in enumerate(birds):
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)


        base.move()
        draw_window(win, birds, pipes, base, score, kGen)




def run(configPath):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                configPath)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)


if __name__ == "__main__":
    localDir = os.path.dirname(__file__)
    configPath = os.path.join(localDir, "config-feedforward.txt")
    run(configPath)