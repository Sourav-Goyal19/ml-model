from manim import *

class LinearRegression(Scene):
    def construct(self):
        # Scene 1: Introduction to Scatter Plots
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="Temperature", y_label="Ice Cream Sales")
        title = Tex("Scatter Plots and Linear Regression").scale(1.2).to_edge(UP)
        self.play(Create(axes), Create(axes_labels), Write(title))

        data_points = VGroup()
        points = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), (10, 8)]
        for x, y in points:
            point = Dot(axes.c2p(x, y), color=BLUE)
            data_points.add(point)
            self.play(Create(point), run_time=0.5)
        self.wait(1)

        # Scene 2: Introducing the "Best-Fit" Line
        initial_line = Line(axes.c2p(0, 0), axes.c2p(10, 10), color=RED)
        best_fit_line = Line(axes.c2p(0, 1.5), axes.c2p(10, 8.5), color=RED)
        self.play(Create(initial_line))
        self.play(Transform(initial_line, best_fit_line), run_time=3)
        self.wait(1)

        # Scene 3: Visualizing Residuals
        residuals = VGroup()
        for x, y in points:
            point = data_points[points.index((x,y))]
            residual = Line(point.get_center(), best_fit_line.get_closest_point(point.get_center()), color=YELLOW)
            residuals.add(residual)
            self.play(Create(residual), run_time=0.5)
        self.wait(1)

        # Scene 4: The Equation of the Line
        equation = MathTex("y = mx + c").next_to(axes, RIGHT)
        m_highlight = SurroundingRectangle(equation[2:3], color=YELLOW)
        c_highlight = SurroundingRectangle(equation[5:6], color=YELLOW)
        self.play(Write(equation), Create(m_highlight), Create(c_highlight))
        self.wait(2)

        # Scene 5: Impact of Data Changes and Outliers
        outlier = Dot(axes.c2p(10, 2), color=BLUE)
        new_best_fit_line = Line(axes.c2p(0, 2.5), axes.c2p(10, 7.5), color=RED)
        self.play(Create(outlier))
        self.play(Transform(best_fit_line, new_best_fit_line), run_time=2)
        self.wait(2)

        # Scene 6: Conclusion and Summary
        summary = Tex("Linear Regression: Modeling Relationships").scale(1.2).to_edge(UP)
        self.play(TransformFromCopy(title, summary), *[FadeOut(mob) for mob in [axes, axes_labels, data_points, outlier, best_fit_line, residuals, equation, m_highlight, c_highlight]], run_time=1)
        self.wait(2)