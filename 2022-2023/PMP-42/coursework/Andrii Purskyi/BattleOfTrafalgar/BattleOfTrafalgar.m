% red = Side('Британсткий флот', 13, 0.36);
% blue = Side('Франко-іспанський флот', 3, 0.36);
% red_replacements = [0.5, 14];
% blue_replacements = [0.52, 17; 2.6, 13];
% 
% battle = Battle('Трафальгарська битва згідно з планом Нельсона', blue, red, 5, 0.01, blue_replacements, red_replacements);
% battle = battle.resolve();
% battle.plot();
% 



% classdef Side
%     properties
%         name
%         strength
%         coefficient
%     end
%     
%     methods
%         function obj = Side(name, strength, coefficient)
%             obj.name = name;
%             obj.strength = strength;
%             obj.coefficient = coefficient;
%         end
%         
%         function ap = attack_power(obj)
%             ap = obj.strength * obj.coefficient;
%         end
%         
%         function obj = damage(obj, amount)
%             obj.strength = obj.strength - min(amount, obj.strength);
%         end
%     end
% end



classdef Battle
    properties
        name
        blue
        red
        duration
        precision
        blue_replacements
        red_replacements
        blue_plot
        red_plot
        time
    end
    
    methods
        function obj = Battle(name, blue, red, duration, precision, blue_replacements, red_replacements)
            obj.name = name;
            obj.blue = blue;
            obj.red = red;
            obj.duration = duration;
            obj.precision = precision;
            obj.blue_replacements = blue_replacements;
            obj.red_replacements = red_replacements;
            obj.blue_plot = zeros(1, round(duration / precision));
            obj.red_plot = zeros(1, round(duration / precision));
            obj.time = zeros(1, round(duration / precision));
        end
        
       function obj = resolve(obj)
    if ~isempty(obj.blue_replacements)
        for i = 1:size(obj.blue_replacements, 1)
            replacements = obj.blue_replacements(i, :);
            index = round(replacements(1) / obj.precision) + 1;
            obj.blue_plot(index) = obj.blue_plot(index) + replacements(2);
        end
    end

    if ~isempty(obj.red_replacements)
        for i = 1:size(obj.red_replacements, 1)
            replacements = obj.red_replacements(i, :);
            index = round(replacements(1) / obj.precision) + 1;
            obj.red_plot(index) = replacements(2);
        end
    end

    obj.blue_plot(1) = obj.blue.strength;
    obj.red_plot(1) = obj.red.strength;
    obj.time(1) = 0;

    for i = 1:(round(obj.duration / obj.precision) - 1)
        obj.blue_plot(i+1) = obj.blue_plot(i+1) + max(0, obj.blue_plot(i) - obj.precision * obj.red_plot(i) * obj.red.coefficient);
        obj.red_plot(i+1) = obj.red_plot(i+1) + max(0, obj.red_plot(i) - obj.precision * obj.blue_plot(i) * obj.blue.coefficient);
        obj.time(i+1) = obj.time(i) + obj.precision;
    end
end

        
        function plot(obj)
            figure
            hold on
            plot(obj.time, obj.blue_plot, 'b', 'LineWidth', 2)
            plot(obj.time, obj.red_plot, 'r', 'LineWidth', 2)
            xlabel('Час')
            ylabel('Чисельність')
            title(obj.name)
            legend({obj.blue.name, obj.red.name}, 'Location', 'best')
            hold off
        end
    end
end


%    scenario of simple lan2

% 
% red = Side('Британсткий флот', 27, 0.25);
% blue = Side('Франко-іспанський флот', 33, 0.25);
% battle = Battle('Трафальгарська битва теоретичний приклад', blue, red, 5, 0.01, [], []);
% battle = battle.resolve();
% battle.plot();