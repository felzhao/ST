## Copyright (C) 2019 felzhao
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} plotData (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: felzhao <felzhao@ZZ0019185N0>
## Created: 2019-04-17

plotData is the command-line function:

function plotData (X, y)
  figure;
  hold ('on');
  pos = find (y == 1);
  neg = find (y == 0);
  plot (X (pos, 1), X (pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
  plot (X (neg, 1), X (neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
  hold ('off');
endfunction

