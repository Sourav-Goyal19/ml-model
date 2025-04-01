from manim import *

class LinearRegression(Scene):
    def construct(self):
        # Scene 1: Introduction - Scatter Plot Generation
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 8, 1],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="Temperature", y_label="Ice Cream Sales")
        data_points = VGroup()
        data = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 5), (9, 6), (10, 7)]
        for x, y in data:
            point = Dot(axes.c2p(x, y), color=BLUE)
            data_points.add(point)
            self.play(Create(point), run_time=0.5)
        self.play(Create(axes), Create(axes_labels))
        self.wait(1)
        self.play(Write(Text("Let's say we have data showing ice cream sales and temperature. Notice the general upward trend – as temperature increases, so do sales.").to_edge(UP)))

        # Scene 2: Introducing the Line of Best Fit
        line = Line(start=axes.c2p(0, 0), end=axes.c2p(10, 10), color=RED)  #Initial line
        best_fit_line = Line(start=axes.c2p(0, 1), end=axes.c2p(10, 6), color=RED) #Actual best fit line

        self.play(Create(line), run_time=0.5)
        self.play(Transform(line, best_fit_line), run_time=2)
        self.play(FadeOut(line))
        self.play(Create(best_fit_line))

        self.wait(1)
        self.play(Write(Text("Linear regression aims to find the 'line of best fit' – a line that best represents the overall trend in our data. Notice how the line is adjusting to minimize the distances between itself and each data point.").to_edge(UP)))

        # Scene 3: Error Representation
        residuals = VGroup()
        for x, y in data:
            point = axes.c2p(x, y)
            closest_point = best_fit_line.get_closest_point(point)
            error = Line(point, closest_point, color=RED)
            residuals.add(error)
            self.play(Create(error), run_time=0.5)
        self.wait(1)
        self.play(Write(Text("The vertical distances between the points and the line represent the error. Linear regression aims to minimize the sum of the *squared* errors. Squaring ensures that both positive and negative errors contribute positively to the total error.").to_edge(UP)))

        # Scene 4: Addressing Misconceptions
        bad_line = VMobject()
        bad_line.set_points_as_corners([axes.c2p(x, y) for x, y in data])
        bad_line.set_color(RED)
        bad_errors = VGroup(*[Line(axes.c2p(x, y), axes.c2p(x, bad_line.get_closest_point(axes.c2p(x, y))[1]), color=RED) for x, y in data])

        self.play(Create(bad_line), *[Create(error) for error in bad_errors], run_time=1)
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in [bad_line, bad_errors]])
        self.wait(1)
        self.play(Write(Text("It's important to note that the line doesn't need to pass through every point. Trying to do so (overfitting) can lead to a model that doesn't generalize well to new data. Our goal is a line that captures the overall trend.").to_edge(UP)))

        # Scene 5: Slope and Intercept
        equation = MathTex("y = mx + c", font_size=36).to_edge(UP)
        m_highlight = SurroundingRectangle(equation[2:3], color=YELLOW)
        c_highlight = SurroundingRectangle(equation[5:6], color=YELLOW)

        self.play(Write(equation), run_time=1)
        self.play(Create(m_highlight), Create(c_highlight))
        self.wait(1)
        self.play(Write(Text("The equation of our line is y = mx + c, where 'm' is the slope (representing the relationship between x and y), and 'c' is the y-intercept (the value of y when x is zero).").to_edge(UP)))
        self.play(FadeOut(m_highlight), FadeOut(c_highlight))

        # Scene 6: Limitations of Linear Regression
        nonlinear_points = VGroup(*[Dot(axes.c2p(x, x**2), color=BLUE) for x in range(1,11)])
        nonlinear_line = Line(start=axes.c2p(0,0), end=axes.c2p(10, 100), color=RED)
        nonlinear_errors = VGroup(*[Arrow(point.get_center(), nonlinear_line.get_closest_point(point.get_center()), buff=0, color=RED) for point in nonlinear_points])

        self.play(*[FadeOut(mob) for mob in [axes, axes_labels, data_points, best_fit_line, residuals, equation]])
        axes2 = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 100, 10],
            x_length=6,
            y_length=4,
            axis_config={"include_numbers": True},
        )
        axes_labels2 = axes2.get_axis_labels(x_label="X", y_label="Y")
        self.play(Create(axes2), Create(axes_labels2), Create(nonlinear_points))
        self.play(Create(nonlinear_line), *[Create(error) for error in nonlinear_errors])
        self.wait(1)
        self.play(Write(Text("Linear regression is powerful, but it's crucial to remember its limitations. It only works well for linear relationships. When the data exhibits a curve, a straight line simply won't capture the trend accurately.").to_edge(UP)))


        # Scene 7: Conclusion
        summary = Text("In summary, linear regression helps us find the best-fitting straight line to represent a linear relationship in our data. Understanding its strengths and limitations is essential for applying it effectively.").to_edge(UP)
        self.play(*[FadeOut(mob) for mob in [nonlinear_line, *nonlinear_errors, nonlinear_points, axes2, axes_labels2]])
        self.play(TransformFromCopy(equation, summary), run_time=1)
        self.wait(2)