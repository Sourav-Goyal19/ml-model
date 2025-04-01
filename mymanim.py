from manim import *

class LinearRegression(Scene):
    def construct(self):
        # Scene 1: Introduction to Scatter Plots
        plane = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"include_numbers": True},
        )
        self.play(Create(plane), run_time=0.5)

        dots = []
        data_points = [(1, 1), (2, 2.5), (3, 3), (4, 4.5), (5, 5), (6, 6.5), (7, 7), (8, 8.5), (9, 9), (10, 9.5)]
        for x, y in data_points:
            dot = Dot(plane.coords_to_point(x, y), color=BLUE)
            dots.append(dot)
            self.play(Create(dot), run_time=0.2)

        self.wait(1)

        #Scatter plot with no correlation
        dots2 = []
        data_points2 = [(1, 9), (2, 1), (3, 7), (4, 3), (5, 5), (6, 8), (7, 2), (8, 6), (9, 4), (10, 10)]
        for x, y in data_points2:
            dot = Dot(plane.coords_to_point(x, y), color=BLUE)
            dots2.append(dot)
        self.play(*[Transform(dots[i],dots2[i]) for i in range(len(dots))], run_time=1)
        self.wait(1)

        #Scatter plot with negative correlation
        dots3 = []
        data_points3 = [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1), (10, 0)]
        for x, y in data_points3:
            dot = Dot(plane.coords_to_point(x, y), color=BLUE)
            dots3.append(dot)
        self.play(*[Transform(dots2[i],dots3[i]) for i in range(len(dots2))], run_time=1)
        self.wait(1)

        self.play(*[FadeOut(dot) for dot in dots3], run_time=0.5)

        #Scene 2: Introducing the Line of Best Fit
        self.play(*[Create(dot) for dot in dots], run_time=0.5)
        line_best = Line(plane.coords_to_point(0,0), plane.coords_to_point(10,10), color=RED)
        self.play(Create(line_best), run_time=0.5)

        #Animation of line moving to best fit (simplified for demonstration)
        target_line = Line(plane.coords_to_point(0, 0.5), plane.coords_to_point(10, 10), color=RED)
        self.play(Transform(line_best, target_line), run_time=1)


        # Scene 3: Minimizing the Sum of Squared Errors
        # (Simplified visualization - actual SSE calculation omitted for brevity)

        # Scene 4: Correlation vs. Causation (Simplified)
        self.wait(1)

        # Scene 5: Slope and Intercept
        equation = MathTex("y = mx + c").to_edge(UP)
        self.play(Write(equation), run_time=0.5)
        
        # (Simplified - no interactive sliders for brevity)

        # Scene 6: Model Evaluation: R-squared (Simplified)
        r_squared = MathTex("R^2 = 0.95").next_to(equation, DOWN)
        self.play(Write(r_squared), run_time=0.5)
        self.wait(1)

        # Scene 7: Conclusion and Summary
        self.play(FadeOut(equation), FadeOut(r_squared), run_time=0.5)
        self.wait(1)